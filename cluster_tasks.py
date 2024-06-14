import os

import coiled
import luigi

import const


def get_cluster():
    return coiled.Cluster(
        name=const.CLUSTER_NAME,
        n_workers=const.START_WORKERS,
        worker_vm_types=['m7a.medium'],
        scheduler_vm_types=['m7a.medium'],
        environ={
            'CLIMATE_ACCESS_KEY': os.environ['CLIMATE_ACCESS_KEY'],
            'CLIMATE_ACCESS_SECRET': os.environ['CLIMATE_ACCESS_SECRET']
        },
        # worker_options={'nthreads': 1}
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
