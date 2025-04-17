from dask.distributed import Client, PipInstall
from dask_gateway import Gateway, GatewayCluster


class Cluster(GatewayCluster):
    """A dask-gateway cluster allowing one cluster per user.
    Instead of always creating new clusters, connect to a previously running
    user cluster, and attempt to limit users to a single cluster.

    Methods
    -------
    get_client(set_as_default=True) -> Client
        Get a dask client connected to the cluster.

    Examples
    --------
    >>> from climakitae.util.cluster import Cluster
    >>> cluster = Cluster() # Create cluster
    >>> cluster.adapt(minimum=0, maximum=8) # Specify the number of workers to use
    >>> client = cluster.get_client()
    >>> cluster # Output cluster information

    """

    extra_packages = ["git+https://github.com/cal-adapt/climakitae.git"]

    def get_client(self, set_as_default: bool = True) -> Client:
        """Get client

        Returns
        -------
        distributed.client.Client
        """
        clusters = self.gateway.list_clusters()
        if clusters:
            cluster = self.gateway.connect(clusters.pop().name, shutdown_on_close=True)
            for c in clusters:
                self.gateway.stop_cluster(c.name)
            client = cluster.get_client()
        else:
            client = super().get_client(set_as_default)
        plugin = PipInstall(packages=self.extra_packages)
        client.register_worker_plugin(plugin)
        return client
