from dask_gateway import Gateway, GatewayCluster


class Cluster(GatewayCluster):
    """A dask-gateway cluster allowing one cluster per user.

    Instead of always creating new clusters, connect to a previously running
    user cluster, and attempt to limit users to a single cluster.
    """

    def get_client(self, set_as_default=True):
        clusters = self.gateway.list_clusters()
        if clusters:
            cluster = self.gateway.connect(clusters.pop().name, shutdown_on_close=True)
            for c in clusters:
                self.gateway.stop_cluster(c.name)
            return cluster.get_client()
        return super().get_client(set_as_default)
