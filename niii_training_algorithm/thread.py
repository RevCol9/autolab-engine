import ctypes
import inspect
import time
from threading import Thread
import sys

# def _async_raise(tid, exctype):
#     tid = ctypes.c_long(tid)
#     if not inspect.isclass(exctype):
#         exctype = type(exctype)
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
#     print(res)
#     print("#############3")
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")

# def stop_thread(thread):
#     print(f'\nt -> {thread}')
#     print(f'\nthread.ident -> {thread.ident}')
#     _async_raise(thread.ident, SystemExit)
#     thread.join()

# def func1():
#     while True:
#         try:
#             print(f'func1')
#             time.sleep(1)
#         except SystemExit:
#             print("!!!!")
#             sys.exit()
#         except:
#             print(traceback.format_exc())

# i = Thread(target=func1, args=())
# i.start()
# time.sleep(3)
# print(f'线程{i}的状态{i.is_alive()}, 线程{i}的名字{i.name}, 线程的方法{i.ident}')
# print(type(i.name), i.name)
# stop_thread(i)


class TrainThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def _async_raise(self,exctype):
        tid = ctypes.c_long(self.ident)
        print(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        print(res)
        print("#############")
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self):
        print(f'\nthread.ident -> {self.ident}')
        self._async_raise(SystemExit)
        self.join()

    # def kill(self):
    #     print("kill")
    #     self._raise_exc(SystemExit)

    def run(self):
        print(f'线程的状态{self.is_alive()}, 线程的名字{self.name}, 线程的方法{self.ident}')
        print(type(self.name), self.name)
        n=10
        while n>0:
            n = n-1
            print(n)
            time.sleep(1)

        self.stop_thread()

my_thread = TrainThread()
my_thread.start()