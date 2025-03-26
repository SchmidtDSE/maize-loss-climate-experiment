"""Tasks for spinning up and turning off the cluster.

Tasks for spinning up and turning off the cluster where behavior can be modified through environment
variables: USE_AWS, AWS_ACCESS_KEY, AWS_ACCESS_SECRET, and SOURCE_DATA_LOC.

License:
    BSD
"""

import os

import luigi

import const

FORCE_ENV = True


class SimulatedDaskCluster:
    """Adapter which pretends to be a remote cluster but executes using local Dask distributed.

    Adapter which pretends to be a remote cluster but executes using Dask distributed, allowing for
    execution of the pipeline optionally without remote machines.
    """

    def __init__(self):
        """Create a handle for a cluster that has not yet started."""
        self._cluster = None

    def get_client(self):
        """Get the cluster client, starting the cluster if needed.

        Returns:
            The cluster client.
        """
        import dask.distributed

        if self._cluster is None:
            self._cluster = dask.distributed.Client()

        return self._cluster

    def close(self, force_shutdown=True):
        """Close this cluster.

        Args:
            force_shutdown: Flag indicating if the the cluster should be hard-terminated if needed.
                True if yes and false if no (which may leave the cluster running). Defaults to true.
        """
        if self._cluster is not None:
            self._cluster.close()

    def adapt(self, minimum=10, maximum=450):
        """Indicate the minimum and maximum resources usable by this cluster.

        Indicate the minimum and maximum resources usable by this cluster, ignored as non-applicable
        by the local cluster.

        Args:
            minimum: The minimum number of machines.
            maximum: The maximum number of machines.
        """
        pass


simulated_cluster = SimulatedDaskCluster()


def get_cluster(machine_type='m7a.medium'):
    """Get the pipeline cluster or start it if it is not running.

    Returns:
        Cluster after requesting it start or SimulatedDaskCluster if using local.
    """
    import coiled

    using_local = os.environ['USE_AWS'] == '0'
    if using_local:
        return simulated_cluster
    else:
        if FORCE_ENV:
            pip = [
                "bokeh!=3.0.*,>=2.4.2",
                "boto3~=1.34.65",
                "coiled~=1.28.0",
                "dask~=2024.3.1",
                "fiona~=1.10.1",
                "geolib~=1.0.7",
                "geotiff~=0.2.10",
                "imagecodecs~=2024.1.1",
                "keras~=3.1.1",
                "libgeohash~=0.1.1",
                "luigi~=3.5.0",
                "more-itertools~=10.5.0",
                "numpy~=1.26.4",
                "pandas~=2.2.2",
                "pathos~=0.3.2",
                "requests~=2.32.0",
                "scipy~=1.12.0",
                "shapely~=2.0.3",
                "tensorflow~=2.16.1",
                "toolz~=0.12.1"
            ]
            coiled.create_software_environment(
                name="maize-env",
                pip=pip,
                include_local_code=True
            )
            return coiled.Cluster(
                name=const.CLUSTER_NAME,
                n_workers=const.START_WORKERS,
                software='maize-env',
                worker_vm_types=[machine_type],
                scheduler_vm_types=[machine_type],
                environ={
                    'AWS_ACCESS_KEY': os.environ.get('AWS_ACCESS_KEY', ''),
                    'AWS_ACCESS_SECRET': os.environ.get('AWS_ACCESS_SECRET', ''),
                    'SOURCE_DATA_LOC': os.environ.get('SOURCE_DATA_LOC', '')
                }
            )
        else:
            return coiled.Cluster(
                name=const.CLUSTER_NAME,
                n_workers=const.START_WORKERS,
                worker_vm_types=[machine_type],
                scheduler_vm_types=[machine_type],
                environ={
                    'AWS_ACCESS_KEY': os.environ.get('AWS_ACCESS_KEY', ''),
                    'AWS_ACCESS_SECRET': os.environ.get('AWS_ACCESS_SECRET', ''),
                    'SOURCE_DATA_LOC': os.environ.get('SOURCE_DATA_LOC', '')
                }
            )


class StartClusterTask(luigi.Task):
    """Task to start the cluster."""

    def output(self):
        """Indicate where status should be written.

        Returns:
            LocalTarget where status should be written.
        """
        return luigi.LocalTarget(const.get_file_location('cluster_start.txt'))

    def run(self):
        """Run this step to start the cluster.

        Run this step to start the cluster, writing the status message with the cluster name and
        dashboard link to the output file.
        """
        cluster = get_cluster(machine_type='m7a.large')
        client = cluster.get_client()

        with self.output().open('w') as f:
            template_vals = (const.CLUSTER_NAME, client.dashboard_link)
            f.write('%s opened at %s' % template_vals)


class StartBigClusterTask(luigi.Task):
    """Task to start a bigger cluster."""

    def output(self):
        """Indicate where status should be written.

        Returns:
            LocalTarget where status should be written.
        """
        return luigi.LocalTarget(const.get_file_location('cluster_big_start.txt'))

    def run(self):
        """Run this step to start the cluster.

        Run this step to start the cluster, writing the status message with the cluster name and
        dashboard link to the output file.
        """
        cluster = get_cluster()
        client = cluster.get_client()

        with self.output().open('w') as f:
            template_vals = (const.CLUSTER_NAME, client.dashboard_link)
            f.write('%s opened at %s' % template_vals)


class EndClusterTask(luigi.Task):
    """Abstract base class for tasks terminate the cluster."""

    def requires(self):
        """Get the task that should be completed prior to termination.

        Returns:
            The task that should complete prior to spinning down the cluster.
        """
        return self.get_prereq()

    def output(self):
        """Get the location where status should be written.

        Returns:
            LocalTarget where status can be written.
        """
        task_name = self.get_task_name()
        return luigi.LocalTarget(const.get_file_location('%s.txt' % task_name))

    def run(self):
        """Terminate the cluster and write out status to disk."""
        cluster = get_cluster()

        cluster.close(force_shutdown=True)

        with self.output().open('w') as f:
            f.write(const.CLUSTER_NAME + ' closed.')

    def get_prereq(self):
        """Get the task that should be completed prior to termination.

        Get the task that should be completed prior to termination, an abstract method to be
        completed by implementor.

        Returns:
            The task that should complete prior to spinning down the cluster.
        """
        raise NotImplementedError('Use implementor.')
