import hydra
import numpy as np
import torch
import yaml
from tqdm import tqdm
import sklearn.metrics
from src.utils import instantiate_callbacks

@hydra.main(version_base=None, config_path='configs/', config_name='config.yaml')
def main(cfg):
    print('Instantiating datamodule \n')
    datamodule = hydra.utils.instantiate(cfg['datamodule'])

    print('Instantiating backbone \n')
    backbone = torch.nn.Sequential(
        hydra.utils.instantiate(cfg['backbone']),
        torch.nn.Sigmoid()
    )

    print('Instantiating optimizer \n')
    optimizer = hydra.utils.instantiate(cfg['optimizer'], params=backbone.parameters())

    print('Instantiating scheduler \n')
    scheduler = hydra.utils.instantiate(cfg['scheduler'], optimizer=optimizer)

    print('Instantiating loss and metric functions \n')
    loss_function = hydra.utils.instantiate(cfg['loss'])
    metric = hydra.utils.instantiate(cfg['metric'])

    print('Instantiating model \n')
    model = hydra.utils.instantiate(
        cfg['model'], 
        backbone=backbone, 
        loss_function=loss_function, 
        metric=metric, 
        optimizer=optimizer,
        scheduler=scheduler
    )

    model.eval()

    model = model.load_from_checkpoint(
        cfg['best_model_checkpoint'],
        backbone=backbone, 
        loss_function=loss_function, 
        metric=metric, 
        optimizer=optimizer,
        scheduler=scheduler
    )

    print('Instantiating callbacks \n')
    callbacks = instantiate_callbacks(cfg['callbacks'])

    print('Instantiating trainer \n')
    trainer = hydra.utils.instantiate(cfg['trainer'], callbacks=callbacks, logger=None)

    print('Computing test metric \n')
    metric = trainer.test(model, datamodule=datamodule, ckpt_path=cfg['best_model_checkpoint'])

    print('Computing AUROC')
    datamodule.prepare_data()
    datamodule.setup('test')
    test_ds = datamodule.test_dataset
    test_dl = datamodule.test_dataloader()

    # Predictions
    outputs = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            images, gts = batch["image"], batch["mask"]
            gts = torch.where(gts > 0, 1, 0).astype(torch.float32)
            preds = model(images)
            preds = torch.where(preds > 0.5, True, False).astype(torch.float32)
            outputs.append(preds.numpy())

    outputs = np.concatenate(outputs)
    
    # GTs
    gts = np.concatenate([test_ds[i]['mask'].numpy()[None, ...] for i in range(len(test_ds))])

    aucs = []
    for i in range(len(outputs)):
        pred = outputs[i][0, ...]
        gt = gts[i][0, ...]
        auc = sklearn.metrics.roc_auc_score(gt.flatten(), pred.flatten())
        aucs.append(auc)

    auroc = float(np.round(np.mean(aucs), 3))

    performances = {
        'dice': round(metric[0]['test_metric'], 3),
        'auroc': auroc
    }

    # Save to YAML
    with open('performances.yaml', 'w') as file:
        yaml.dump(performances, file)

if __name__ == "__main__": 
    main()