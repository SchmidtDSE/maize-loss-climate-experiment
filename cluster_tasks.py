import os

import coiled
import dask.distributed
import luigi

import const


class SimulatedDaskCluster:

    def __init__(self):
        self._cluster = None

    def get_client(self):
        if self._cluster is None:
            self._cluster = dask.distributed.Client()

        return self._cluster

    def close(self, force_shutdown=True):
        if self._cluster is not None:
            self._cluster.close()

    def adapt(self, minimum=10, maximum=500):
        pass


simulated_cluster = SimulatedDaskCluster()


def get_cluster():
    using_local = os.environ['USE_AWS'] == '0'
    if using_local:
        return simulated_cluster
    else:
        return coiled.Cluster(
            name=const.CLUSTER_NAME,
            n_workers=const.START_WORKERS,
            worker_vm_types=['m7a.medium'],
            scheduler_vm_types=['m7a.medium'],
            environ={
                'AWS_ACCESS_KEY': os.environ.get('AWS_ACCESS_KEY', ''),
                'AWS_ACCESS_SECRET': os.environ.get('AWS_ACCESS_SECRET', ''),
                'SOURCE_DATA_LOC': os.environ.get('SOURCE_DATA_LOC', '')
            }
        )


class StartClusterTask(luigi.Task):

    def output(self):
        return luigi.LocalTarget(const.get_file_location('cluster_start.txt'))

    def run(self):
        cluster = get_cluster()
        client = cluster.get_client()

        with self.output().open('w') as f:
            template_vals = (const.CLUSTER_NAME, client.dashboard_link)
            f.write('%s opened at %s' % template_vals)


class EndClusterTask(luigi.Task):

    def requires(self):
        return self.get_prereq()

    def output(self):
        task_name = self.get_task_name()
        return luigi.LocalTarget(const.get_file_location('%s.txt' % task_name))

    def run(self):
        cluster = get_cluster()

        cluster.close(force_shutdown=True)

        with self.output().open('w') as f:
            f.write(const.CLUSTER_NAME + ' closed.')

    def get_prereq(self):
        raise NotImplementedError('Use implementor.')
