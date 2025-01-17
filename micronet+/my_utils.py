import time


def report_time_and_speed(t_begin, epoch, epochs, len_train_loader):
    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / len_train_loader
    eta = speed_epoch * epochs - elapse_time
    print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s\n".format(
        elapse_time, speed_epoch, speed_batch, eta))