from setuptools import setup, find_packages


def get_requirements():
    with open('yolov5/requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='yolov5',
    version='1.0.0',
    url='https://github.com/ultralytics/yolov5',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=get_requirements(),
    include_package_data=True,
    entry_points={'console_scripts': [
        "yolov5=yolov5.cli:app",
    ]}
)
