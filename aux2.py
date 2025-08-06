import config as cfg

cfg.stored_info.email_content['feedback'] =  'a'

def get_user_data():
    return cfg.stored_info.email_content
get_user_data.__doc__ = cfg.ToolDocstrings.get_user_data

print(get_user_data())