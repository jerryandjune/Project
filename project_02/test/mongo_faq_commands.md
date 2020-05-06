## 启用和重启mongo服务
sudo service mongod start
sudo service mongod restart

## 修改mongo配置文件
sudo nano /etc/mongod.conf

## 超级用户建立 （需要在配置文件中关闭验证）
1. 进入mongo
mongo
2. 进入admin数据库
use admin
3. 建立超级用户
db.createUser(
  {
    user: "jack",
    pwd:  'jackPasswd',
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
4. 测试超级用户
db.auth('jack','jackPasswd')

## 具体数据库授权 或者 创建用户
1. 用超级用户登录mongo
mongo --port 27017  --authenticationDatabase "admin" -u "jack" -p 'jackPasswd'
2. 进入需要授权的数据库
use NlpBallisticAnalysis
3. 授权
db.createUser(
  {
    user: "jack",
    pwd:  'jackPasswd',   // or cleartext password
    roles: [ { role: "readWrite", db: "NlpBallisticAnalysis" } ]
  }
)
4. 测试用户
db.auth('jack','jackPasswd')