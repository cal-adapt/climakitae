"""Wrapper for creating a Dask Gateway Cluster"""

from dask_gateway import Gateway, GatewayCluster


class Cluster(GatewayCluster):
    """A dask-gateway cluster allowing one cluster per user.
    Instead of always creating new clusters, connect to a previously running
    user cluster, and attempt to limit users to a single cluster.

    Examples
    --------
    >>> from climakitae.cluster import Cluster
    >>> cluster = Cluster() # Create cluster
    >>> cluster.adapt(minimum=0, maximum=8) # Specify the number of workers to use
    >>> client = cluster.get_client()
    >>> cluster # Output cluster information

    """

    def get_client(self, set_as_default=True):
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
            return cluster.get_client()
        return super().get_client(set_as_default)
