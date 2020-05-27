## Flask文件配置
1. 复制新闻文件和model文件到Flask\app\Static文件夹
2. 配置Flask\app\models\Config.py中的新闻和model名字

## Flask vscode 调试：
1. 在调试界面选择'Falsk App'配置，F5即可运行
2. 终端打开Flask文件夹，执行python -m flask run

## Flask 运行：
1. app 文件夹上传到web服务器 Ubuntu 目录
2. command to run：
    sudo systemctl start flaskapp
 to check running status:
    sudo systemctl status flaskapp

the service contents for flaskapp
    /etc/systemd/system/flaskapp.service

sudo python3 -m flask run -h 0.0.0.0 -p 80

===============
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
PIDFile=/run/gunicorn/pid
User=ubuntu
Group=www-data
RuntimeDirectory=gunicorn
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/.local/bin/gunicorn --pid /run/gunicorn/pid   --workers 3 \
           --bind 0.0.0.0:5000 --workers 3 "app:create_app('')"
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target
============