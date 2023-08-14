import paho.mqtt.publish as publish

# 读取文件内容
with open("./datasets/helmon/test.jpg", "rb") as f:
    file_data = f.read()

# 发布文件流数据
publish.single("/petrochemical/Service/Command", payload=file_data, qos=1, retain=False, hostname="localhost",
               port=1883)
