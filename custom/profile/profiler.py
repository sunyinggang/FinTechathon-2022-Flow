from functools import wraps
import cProfile
import pstats
import os


def profiler(statfile="/root/output.stat", reportfile="/root/report.txt"):
    """
    对函数进行性能探测的装饰器
    statfile: 输出stat文件的路径，stat文件为二进制文件
    reportfile: 输出文本形式的性能报告的路径

    P.S. 产生图片形式的可视化分析报告
    命令：gprof2dot -f pstats 输入文件.stat | dot -Tpng -o 输出图片.png
    需要安装gprof2dot和graphviz:
    > pip install gprof2dot
    > yum install graphviz
    """
    def profiler_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # Start profiling
            pr = cProfile.Profile()
            pr.enable()

            ret = func(*args, **kwargs)

            # Finish profiling
            pr.disable()
            s = pstats.Stats(pr)
            s.dump_stats(statfile)

            # Output profile
            with open(reportfile, 'w') as stream:
                stats = pstats.Stats(statfile, stream=stream)
                stats.print_stats()

            return ret
        return wrapped_function
    return profiler_decorator


@profiler()
def helloworld():
    print("Hello")
    print("Hello")
    print("helllo" + " world")


helloworld()
