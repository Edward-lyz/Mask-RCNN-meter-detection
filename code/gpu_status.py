# -*- coding: UTF-8 -*-
import pynvml
import time
import smtplib
from email.mime.text import MIMEText 
from email.utils import formataddr
import datetime
import os

threshold = 8000
mail_host = "smtp.qq.com"
mail_user = "1208870448@qq.com"
mail_pass = "vkvjvkjwnofqgbbi"
sender = "1208870448@qq.com"
receiver = ['1208870448@qq.com']


def mail(info):
    msg = MIMEText(info, 'plain', 'utf-8')
    msg['From'] =formataddr(["From XXX",sender])
    msg['To'] = formataddr(["FK", "edwardlyz@foxmail.com"])
    subject = 'GPU使用情况汇报！'
    msg['Subject'] = subject
    server = smtplib.SMTP_SSL(mail_host, 465)
    server.login(mail_user, mail_pass)
    server.sendmail(mail_user, ["edwardlyz@foxmail.com",], msg.as_string())
    server.quit()
    print("SUCCESS")


def send_mail_2_me():
    pynvml.nvmlInit()
    while 1:
        usage = [0]
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage[0] = int(meminfo.used/1024/1024)
        if int(meminfo.used/1024/1024) < threshold:
            info = '''
            GPU Usage :
                    GPU 0: {}M!
            '''.format(usage[0])
            mail(info)
            time.sleep(60)
        else:
            print("ALL BUSY NOW!!!!")
        
        time.sleep(30)
if __name__=='__main__':
    # 范围时间
    d_time = datetime.datetime.strptime(str(datetime.datetime.now().date())+'9:30', '%Y-%m-%d%H:%M')
    d_time1 =  datetime.datetime.strptime(str(datetime.datetime.now().date())+'21:00', '%Y-%m-%d%H:%M')
    while(1):
        # 当前时间
        n_time = datetime.datetime.now()
        # 判断当前时间是否在范围时间内
        if n_time > d_time and n_time<d_time1:
            send_mail_2_me()
            os.system("python meter_detection.py")
        else:
            break
        