from concurrent.futures import ThreadPoolExecutor
import threading
import time
 
# 定义获取线程return返回测试的方法
def action_a(max):
    my_sum = max + 1
    return my_sum
 
def action_b(max):
    my_sum = max + 100
    time.sleep(3)
    return my_sum
 
# 创建包含2个线程的线程池
pool = ThreadPoolExecutor(max_workers=2)
# 向线程池提交一个任务, 20和10会作为action_a/b()方法的参数
future1 = pool.submit(action_a, 20)
future2 = pool.submit(action_b, 10)
 
# 判断future1线程是否结束---返回False表示该线程未结束，True表示该线程已经结束
print("future1线程的状态:" + str(future1.done()))       #此时future1线程已结束
# 判断future2线程是否结束
print("future2线程的状态:" + str(future2.done()))        #此时future2线程未结束，因为休眠了3秒
 
# 查看future1代表的任务返回的结果，如果线程未运行完毕，会暂时阻塞，等待线程运行完毕后再执行、输出；
print(future1.result())        #此处会直接输出
# 查看future2代表的任务返回的结果
print(future2.result())        #此处会等待3秒，因为方法中休眠了3秒
 
# 关闭线程池
pool.shutdown()