from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

def load_system_prompts():
    try:
        #检查是否有main_prompt_path配置
        system_prompts_path = get_abs_path(prompts_conf["main_prompt_path"])
    except Exception as e:
        logger.error(f"[load_system_prompts]加载系统提示词失败: {str(e)}")
        raise e

    try:
        return open(system_prompts_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]读取系统提示词失败: {str(e)}")
        raise e


def load_rag_prompts():
    try:
        #检查是否有rag_summarize_prompt_path配置
        rag_prompts_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except Exception as e:
        logger.error(f"[load_rag_prompts]加载rag提示词失败: {str(e)}")
        raise e

    try:
        return open(rag_prompts_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]读取rag提示词失败: {str(e)}")
        raise e

def load_report_prompts():
    try:
        #检查是否有report_prompt_path配置
        report_prompts_path = get_abs_path(prompts_conf["report_prompt_path"])
    except Exception as e:
        logger.error(f"[load_report_prompts]加载报告提示词失败: {str(e)}")
        raise e

    try:
        return open(report_prompts_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]读取报告提示词失败: {str(e)}")
        raise e

if __name__ == '__main__':
    print(load_system_prompts())
