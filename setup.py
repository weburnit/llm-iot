from setuptools import setup, find_packages

setup(
    name='aitomic',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fastapi',
        'uvicorn',
        'openai',  # Assuming that you are using OpenAI's GPT-3 model
        'pydantic',
        'httpx'
    ],
)