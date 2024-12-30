huggingface-cli download --token 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 \
--repo-type model \
--resume-download Qwen/Qwen2-VL-7B-Instruct \
--local-dir /home/SENSETIME/zengwang/Downloads/Qwen2-VL-7B-Instruct --local-dir-use-symlinks False


~/ads-cli sync \
/home/SENSETIME/zengwang/Downloads/Qwen2-VL-7B-Instruct \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/ckpt/Qwen2-VL-7B-Instruct


/afs/zengwang/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/ckpt/Qwen2-VL-7B-Instruct \
/afs/zengwang/ckpt/Qwen2-VL-7B-Instruct



~/ads-cli sync \
/home/SENSETIME/zengwang/myprojects/task_define_service/data/video_event \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/data/video_event


/afs/zengwang/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/data/video_event \
/afs/zengwang/projects/task_define_service/data/video_event





apt install tmux
ln -s /afs/zengwang/.ssh ~/

cd  /afs/zengwang/projects/task_define_service/
git clone git@github.com:zengwang430521/Qwen2-VL.git