from setuptools import setup
from torch.utils import cpp_extension as cpp
import datetime

ind = 0

def get_version():
    return f'{datetime.date.today().strftime("%Y.%m.%d")}.{ind}'

while True:
    ind += 1
    try:
        file = open("./versions/cppext_ver_" + get_version() + ".placeholder")
        file.close()
        continue
    except:
        break

with open("./versions/cppext_ver_" + get_version() + ".placeholder", 'w') as file:
    file.write(get_version())

setup(
    name='point_match_cpp',
    version=get_version(),
    ext_modules=[cpp.CppExtension(
        'point_match_cpp', ['point_match/point_match.cpp'], 
        extra_compile_args={"cxx": ["-O3", "-std=c++17", "-DPY_BIND"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)