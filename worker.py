
import tensorflow as tf


def main(_):

    FLAGS = tf.app.flags.FLAGS
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    # define the cluster includes parameter servers and workers
    tf.app.flags.DEFINE_string("ps_hosts", "localhost:2224", "ps hosts separated by ','")
    tf.app.flags.DEFINE_string("worker_hosts", "localhost:2222,localhost:2223", "worker hosts separated by ','")

    tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

    tf.app.run()
