args = dict(
    train=dict(
        data_dir="/data01/dect/DECT-SNUH/preliminary",
        save_dir="/data01/dect/result/baseline_version3",
        epochs=500,
        shuffle=True,
        log_cp=50,
        batch_size=128,
        num_workers=32,
        gpus="01234567",
        num_d_optimize=1,
        num_g_optimize=3,
        label_smoothing_factor=0.9,
        image_save_batch_idx=[2, 4],
        model_init='normal',
        loss_lambda=dict(
            lambda_d_real_loss=1.,
            lambda_d_fake_loss=1.,
            lambda_g_fake_loss=5.,
            lambda_l1_recon_loss=5.,
            lambda_l2_recon_loss=5.,
            lambda_ssim_loss=1.,
        ),
        optimizer_type=dict(
            generator='adam',
            discriminator='adam'
        ),
        learning_rate=dict(
            generator=0.001,
            discriminator=0.001
        ),
        step_size=dict(
            generator=30,
            discriminator=30
        ),
        model_param=dict(
            generator=dict(
                in_channels=1,
                bw_channels=64,
                skip=True,
            ),
            discriminator=dict(
                in_channels=1,
                bw_channels=64,
            )
        ),
        model_name=dict(
            generator="baseline",
            discriminator="baseline"
        ),
        model_path=dict(
            generator=None,
            discriminator=None
        )
    ),
    test=dict(
        data_dir="/data01/dect/DECT-SNUH/preliminary",
        model_dir="/data01/dect/result/baseline/generator.pth",
        model_path=dict(
            generator=None,
            discriminator=None
        )
    )
)

if __name__ == '__main__':


