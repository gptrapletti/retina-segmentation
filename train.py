import hydra
import torch
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

    print('Instantiating callbacks \n')
    callbacks = instantiate_callbacks(cfg['callbacks'])

    print('Instantiating logger \n')
    logger = hydra.utils.instantiate(cfg['logger'])

    print('Instantiating trainer \n')
    trainer = hydra.utils.instantiate(cfg['trainer'], callbacks=callbacks, logger=logger)

    print('Training... \n')
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__": 
    main()