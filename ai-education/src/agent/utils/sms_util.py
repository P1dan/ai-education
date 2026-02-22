# sms_util.py
import json
import random
import string
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from typing import Tuple, Optional
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class AliyunSmsUtil:
    def __init__(self):
        # 从环境变量读取配置
        self.access_key_id = os.getenv("ALIYUN_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
        self.sign_name = os.getenv("ALIYUN_SMS_SIGN_NAME")  # 短信签名
        self.template_code = os.getenv("ALIYUN_SMS_TEMPLATE_CODE")  # 模板Code
        self.region_id = os.getenv("ALIYUN_SMS_REGION_ID", "cn-hangzhou")  # 区域，默认杭州

        # 验证必要配置是否存在
        self._validate_config()

    def _validate_config(self):
        """验证必要的配置是否存在"""
        required_configs = {
            "ALIYUN_ACCESS_KEY_ID": self.access_key_id,
            "ALIYUN_ACCESS_KEY_SECRET": self.access_key_secret,
            "ALIYUN_SMS_SIGN_NAME": self.sign_name,
            "ALIYUN_SMS_TEMPLATE_CODE": self.template_code,
        }

        missing_configs = [key for key, value in required_configs.items() if not value]

        if missing_configs:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_configs)}")

    def send_sms(self, phone_numbers: str, code: str) -> Tuple[bool, str]:
        """
        发送短信验证码
        :param phone_numbers: 接收手机号，支持多个，用逗号分隔
        :param code: 验证码内容
        :return: (success, message)
        """
        try:
            # 创建客户端
            client = AcsClient(
                self.access_key_id,
                self.access_key_secret,
                self.region_id
            )

            # 创建通用请求
            request = CommonRequest()
            request.set_method('POST')
            request.set_domain('dysmsapi.aliyuncs.com')
            request.set_version('2017-05-25')
            request.set_action_name('SendSms')

            # 设置参数
            request.add_query_param('PhoneNumbers', phone_numbers)
            request.add_query_param('SignName', self.sign_name)
            request.add_query_param('TemplateCode', self.template_code)

            # 模板参数：将验证码转为JSON格式
            template_param = json.dumps({'code': code})
            request.add_query_param('TemplateParam', template_param)

            # 发送请求
            response = client.do_action_with_exception(request)
            response_dict = json.loads(response)

            # 判断返回结果
            if response_dict.get('Code') == 'OK':
                return True, "发送成功"
            else:
                return False, response_dict.get('Message', '发送失败')

        except Exception as e:
            return False, str(e)

    @staticmethod
    def generate_code(length: int = 6) -> str:
        """
        生成随机验证码（纯数字）
        :param length: 验证码长度，默认6位
        :return: 生成的验证码
        """
        return ''.join(random.choices(string.digits, k=length))

    @staticmethod
    def generate_code_with_letters(length: int = 6) -> str:
        """
        生成随机验证码（数字+字母）
        :param length: 验证码长度，默认6位
        :return: 生成的验证码
        """
        chars = string.digits + string.ascii_uppercase
        return ''.join(random.choices(chars, k=length))


# 使用示例的依赖注入函数
async def get_sms_util() -> AliyunSmsUtil:
    """
    涉及联网操作
    依赖注入函数，用于FastAPI路由中获取短信工具实例
    """
    return AliyunSmsUtil()