conda activate daocc0813



##运行train.py
export PYTHONPATH=/data5/wangyijun/DAOcc:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/data5/wangyijun/DAOcc/torch_extensions_cache

export MASTER_HOST=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0


CUDA_VISIBLE_DEVICES=3 python tools/train.py \
/data5/wangyijun/DAOcc/configs/nuscenes/occ3d/deprecated/daocc_occ3d_nus_w_mask.yaml \
--run-dir /data5/wangyijun/DAOcc/save \
--model.encoders.camera.backbone.init_cfg.checkpoint \
/data5/wangyijun/DAOcc/configs/htc_r50_backbone.pth \
> /data5/wangyijun/DAOcc/save/train.log 2>&1
