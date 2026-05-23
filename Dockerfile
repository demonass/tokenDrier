FROM python:3.10-slim

WORKDIR /app

# 💡 升级点：加入了安全审计工具 bandit
RUN pip install pytest pytest-html tiktoken pydantic requests regex typing-inspection bandit -i https://pypi.tuna.tsinghua.edu.cn/simple

# 强行触发 tiktoken 实例化以固化权重
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

RUN pip cache purge
