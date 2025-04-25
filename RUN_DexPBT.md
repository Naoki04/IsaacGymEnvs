#### PBT for ShadowHand
```
python isaacgymenvs/train.py task=AllegroHand pbt=pbt_default wandb_activate=True wandb_project=sapg_allegro_hand wandb_group="pbt" wandb_entity="naoki-shitanda" num_envs=24576 headless=True wandb_name="pbt_seed0" seed=0 
```

#### PBT for AllegroHand
```
python isaacgymenvs/train.py task=AllegroHand pbt=pbt_default wandb_activate=True wandb_project=sapg_allegro_hand wandb_group="pbt" wandb_entity="naoki-shitanda" num_envs=24576 headless=True wandb_name="pbt_seed0" seed=0 
```


#### PBT for AllegroKukaLSTM
```
python isaacgymenvs/train.py task=AllegroKukaLSTM task/env=regrasping pbt=pbt_default wandb_activate=True wandb_project=sapg_allegro_kuka_regrasping wandb_group="pbt" wandb_entity="naoki-shitanda" num_envs=24576 headless=True wandb_name="pbt_seed0" seed=0 
```

