import argparse
import os

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from dcdfg.callback import (AugLagrangianCallback, ConditionalEarlyStopping,
                            CustomProgressBar)
from dcdfg.linear_baseline.model import LinearGaussianModel
from dcdfg.lowrank_linear_baseline.model import LinearModuleGaussianModel
from dcdfg.lowrank_mlp.model import MLPModuleGaussianModel
from dcdfg.sachsprotein_data import ProteinInterventionDataset

"""
USAGE:
python -u run_sachsprotein.py --reg-coeff 0.01 --constraint-mode spectral_radius --lr 0.01 --model linear
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data-dir", type=str, default="data/2005_sachs_protein", help="Path to data directory"
    )
    parser.add_argument(
        "--train-samples",
        type=float,
        default=0.8,
        help="Fraction of samples used for training (default is 80% of the total size)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="Number of samples in a minibatch",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=600,
        help="Number of meta gradient steps for training",
    )
    parser.add_argument(
        "--num-fine-epochs", type=int, default=50, help="Number of meta gradient steps for fine-tuning"
    )
    # parser.add_argument("--num-modules", type=int, default=20, help="Number of modules in the model")
    # sweep 1, 2, 3, 4, 5, 10
    parser.add_argument("--num-modules", type=int, default=5, help="Number of modules in the model")
    
    
    # optimization
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimization"
    )
    parser.add_argument(
        "--reg-coeff",
        type=float,
        default=0.1,
        help="Regularization coefficient (lambda)",
    )
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default="exp",
        help="Technique for acyclicity constraint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="Model type: linear|linearlr|mlplr",
    )
    parser.add_argument(
        "--poly", action="store_true", help="Polynomial on linear model"
    )
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()

    # Define file mappings and intervention targets
    file_map = {
        "cd3_cd28": "1. cd3cd28.xls",
        "cd3_cd28_icam2": "2. cd3cd28icam2.xls",
        "cd3_cd28+aktinhib": "3. cd3cd28+aktinhib.xls",
        "cd3_cd28+g0076": "4. cd3cd28+g0076.xls",
        "cd3_cd28+psitect": "5. cd3cd28+psitect.xls",
        "cd3_cd28+u0126": "6. cd3cd28+u0126.xls",
        "cd3_cd28+ly": "7. cd3cd28+ly.xls",
        "pma": "8. pma.xls",
        "b2camp": "9. b2camp.xls",
        "cd3_cd28_icam2+aktinhib": "10. cd3cd28icam2+aktinhib.xls",
        "cd3_cd28_icam2+g0076": "11. cd3cd28icam2+g0076.xls",
        "cd3_cd28_icam2+psit": "12. cd3cd28icam2+psit.xls",
        "cd3_cd28_icam2+u0126": "13. cd3cd28icam2+u0126.xls",
        "cd3_cd28_icam2+ly": "14. cd3cd28icam2+ly.xls",
    }

    intervention_targets = {
        'cd3_cd28': {'type': 'activator', 'targets': ['ZAP70', 'Lck', 'plcg', 'praf', 'pmek', 'Erk', 'PKC']},
        'icam2': {'type': 'activator', 'targets': ['LFA-1']},
        'b2camp': {'type': 'activator', 'targets': ['PKA']},
        'pma': {'type': 'activator', 'targets': ['PKC']},
        'aktinhib': {'type': 'inhibitor', 'targets': ['pakts473']},
        'g0076': {'type': 'inhibitor', 'targets': ['PKC']},
        'psitect': {'type': 'inhibitor', 'targets': ['P38']},
        'u0126': {'type': 'inhibitor', 'targets': ['pmek', 'Erk']},
        'ly': {'type': 'inhibitor', 'targets': ['PI3K', 'pakts473']}
    }

    # Instantiate the dataset
    train_dataset = ProteinInterventionDataset(
        file_map, base_dir=args.data_dir, intervention_targets=intervention_targets, fraction_regimes_to_ignore=0.2, normalize=True
    )
    regimes_to_ignore = train_dataset.regimes_to_ignore
    test_dataset = ProteinInterventionDataset(
        file_map, base_dir=args.data_dir, intervention_targets=intervention_targets, regimes_to_ignore=regimes_to_ignore, load_ignored=True, normalize=True
    )

    nb_nodes = test_dataset.dim

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    identifier = f"out/sachs_m-{args.model}_c-{args.constraint_mode}_f-{args.num_modules}_l-{args.lr}_r-{args.reg_coeff}/"
    os.makedirs(identifier, exist_ok=True)

    # Model selection
    if args.model == "linear":
        model = LinearGaussianModel(
            nb_nodes,
            lr_init=args.lr,
            reg_coeff=args.reg_coeff,
            constraint_mode=args.constraint_mode,
            poly=args.poly,
        )
    elif args.model == "linearlr":
        model = LinearModuleGaussianModel(
            nb_nodes,
            args.num_modules,
            lr_init=args.lr,
            reg_coeff=args.reg_coeff,
            constraint_mode=args.constraint_mode,
        )
    elif args.model == "mlplr":
        model = MLPModuleGaussianModel(
            nb_nodes,
            2,
            args.num_modules,
            16,
            lr_init=args.lr,
            reg_coeff=args.reg_coeff,
            constraint_mode=args.constraint_mode,
        )
    else:
        raise ValueError("couldn't find model")

    logger = WandbLogger(project="DCDI-train-2005-sachs-protein", log_model=True)
    
    # Step 1: augmented Lagrangian training
    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.num_train_epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
    )
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4),
        DataLoader(val_dataset, num_workers=8, batch_size=256),
    )
    wandb.log({"nll_val": model.nlls_val[-1]})
    wandb.finish()

    # Freeze and prune adjacency
    model.module.threshold()
    model.module.constraint_mode = "exp"
    model.gamma = 0.0
    model.mu = 0.0

    # Step 2: Fine-tune weights with frozen model
    logger = WandbLogger(project="DCDI-train-2005-sachs-protein", log_model=True)
    early_stop_2_callback = EarlyStopping(
        monitor="Val/nll", min_delta=1e-6, patience=5, verbose=True, mode="min"
    )
    trainer_fine = pl.Trainer(
        devices=args.num_gpus,#gpus=arg.num_gpus,
        accelerator="cpu",
        max_epochs=args.num_fine_epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[early_stop_2_callback, CustomProgressBar()],
    )
    trainer_fine.fit(
        model,
        DataLoader(train_dataset, batch_size=args.train_batch_size),
        DataLoader(val_dataset, num_workers=2, batch_size=256),
    )

    # EVAL on held-out data
    pred = trainer_fine.predict(
        ckpt_path="best",
        dataloaders=DataLoader(test_dataset, num_workers=8, batch_size=256),
    )
    held_out_nll = np.mean([x.item() for x in pred])
    # TODO: also want i_MAE
    dd = torch.tensor(test_dataset.data.astype('float')).to(dtype=torch.float32)
    dm = torch.tensor(test_dataset.masks.astype(bool)) # .type_as(dd) # not sparse
    held_out_mae = model.mae(dd, dm)

    # Step 3: score adjacency matrix
    model.module.save(identifier)
    pred_adj = model.module.weight_mask.detach().cpu().numpy()
    assert np.equal(np.mod(pred_adj, 1), 0).all()
    print("saved, now evaluating")

    # Step 4: add valid NLL and dump metrics
    pred = trainer_fine.predict(
        ckpt_path="best",
        dataloaders=DataLoader(val_dataset, num_workers=8, batch_size=256),
    )
    val_nll = np.mean([x.item() for x in pred])

    acyclic = int(model.module.check_acyclicity())
    wandb.log(
        {
            "interv_nll": held_out_nll,
            "val nll": val_nll,
            "acyclic": acyclic,
            "n_edges": pred_adj.sum(),
            "interv_mae": held_out_mae,
        }
    )
