import os
os.exitpoint(os.EXITPOINT_ENABLE)
import machine, time, binascii

f = machine.FPIOA()
f.set_function(32, f.UART3_TXD)
f.set_function(33, f.UART3_RXD)
uart = machine.UART(machine.UART.UART3, 115200)

frame = bytes([0xAA, 0x55, 0x00, 0x0A, 0x00, 0x14, 0x01, 0x1E])

n = 0
while True:
    uart.write(frame)
    n = n + 1
    time.sleep_ms(50)
    back = uart.read()
    if back:
        print(n, "收:", binascii.hexlify(back))
    time.sleep_ms(50)
