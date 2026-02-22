# email_util.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr
from typing import Tuple, Optional
import random
import string
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmailUtil:
    def __init__(self):
        # 从环境变量读取配置
        self.smtp_server = os.getenv("EMAIL_SMTP_SERVER")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", 587))
        self.sender_email = os.getenv("EMAIL_SENDER")
        self.sender_password = os.getenv("EMAIL_PASSWORD")  # 或授权码
        self.use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
        self.use_ssl = os.getenv("EMAIL_USE_SSL", "false").lower() == "true"

        # 验证必要配置是否存在
        self._validate_config()

    def _validate_config(self):
        """验证必要的配置是否存在"""
        required_configs = {
            "EMAIL_SMTP_SERVER": self.smtp_server,
            "EMAIL_SENDER": self.sender_email,
            "EMAIL_PASSWORD": self.sender_password,
        }

        missing_configs = [key for key, value in required_configs.items() if not value]

        if missing_configs:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_configs)}")

    def send_verification_email(self, to_email: str, code: str, subject: str = "验证码") -> Tuple[bool, str]:
        email_msg = MIMEMultipart()
        sender_name = "AI教育平台"
        email_msg["From"] = formataddr((sender_name, self.sender_email))
        email_msg["To"] = to_email
        email_msg["Subject"] = Header(subject, "utf-8")  # 显式指定编码更安全

        body = f"您的验证码是：{code}，5分钟内有效。如非本人操作，请忽略此邮件。"
        email_msg.attach(MIMEText(body, "plain", "utf-8"))

        # 使用 as_bytes() 获取字节流
        msg_bytes = email_msg.as_bytes()

        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) if self.use_ssl else smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            if self.use_tls and not self.use_ssl:
                server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, [to_email], msg_bytes)

        return True, "邮件发送成功"


    @staticmethod
    def generate_code(length: int = 6) -> str:
        """
        生成纯数字验证码，虽然已经实现了，但这边可以写一下，如果以后考虑解耦可以用
        :param length: 验证码长度，默认6位
        :return: 字符串形式的验证码
        """
        return ''.join(random.choices(string.digits, k=length))

    @staticmethod
    def generate_code_with_letters(length: int = 6) -> str:
        """
        生成数字+大写字母混合验证码
        :param length: 验证码长度，默认6位
        :return: 字符串形式的验证码
        """
        chars = string.digits + string.ascii_uppercase
        return ''.join(random.choices(chars, k=length))


# FastAPI 依赖注入函数，每次返回一个新实例
# 由于本身不涉及IO操作，用同步的即可
def get_email_util() -> EmailUtil:
    return EmailUtil()

