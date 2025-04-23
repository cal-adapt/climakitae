from unittest.mock import MagicMock, call, patch

import pytest

from climakitae.util.cluster import Cluster


class TestCluster:
    @patch("climakitae.util.cluster.PipInstall")
    @patch("climakitae.util.cluster.Gateway")
    @patch("climakitae.util.cluster.GatewayCluster.__init__", return_value=None)
    @patch("climakitae.util.cluster.GatewayCluster.get_client")
    def test_get_client_no_clusters(
        self,
        mock_super_get_client: MagicMock,
        mock_init: MagicMock,
        mock_gateway_class: MagicMock,
        mock_pip_install: MagicMock,
    ):
        # Setup
        mock_gateway_instance = MagicMock()
        mock_gateway_instance.list_clusters.return_value = []
        mock_gateway_class.return_value = mock_gateway_instance

        # Mock PipInstall creation
        mock_pip_install.return_value = MagicMock()

        cluster = Cluster()
        cluster.gateway = mock_gateway_instance
        mock_client = MagicMock()
        mock_super_get_client.return_value = mock_client

        # Execute
        result = cluster.get_client(set_as_default=True)

        # Verify
        assert result == mock_client
        mock_super_get_client.assert_called_once_with(True)
        mock_client.register_worker_plugin.assert_called_once()

        # Verify PipInstall was called with correct packages
        mock_pip_install.assert_called_once_with(packages=cluster.extra_packages)

    @patch("climakitae.util.cluster.PipInstall")
    @patch("climakitae.util.cluster.Gateway")
    @patch("climakitae.util.cluster.GatewayCluster.__init__", return_value=None)
    @patch("climakitae.util.cluster.GatewayCluster.get_client")
    def test_get_client_one_cluster(
        self,
        mock_super_get_client: MagicMock,
        mock_init: MagicMock,
        mock_gateway_class: MagicMock,
        mock_pip_install: MagicMock,
    ):
        # Setup
        mock_gateway_instance = MagicMock()
        mock_gateway_class.return_value = mock_gateway_instance

        # Mock PipInstall creation
        mock_pip_install.return_value = MagicMock()

        cluster = Cluster()
        cluster.gateway = mock_gateway_instance

        existing_cluster = MagicMock()
        existing_cluster.name = "existing-cluster"
        cluster.gateway.list_clusters.return_value = [existing_cluster]

        mock_connected_cluster = MagicMock()
        mock_client = MagicMock()
        mock_connected_cluster.get_client.return_value = mock_client
        cluster.gateway.connect.return_value = mock_connected_cluster

        # Execute
        result = cluster.get_client()

        # Verify
        assert result == mock_client
        cluster.gateway.connect.assert_called_once_with(
            "existing-cluster", shutdown_on_close=True
        )
        mock_super_get_client.assert_not_called()
        mock_client.register_worker_plugin.assert_called_once()

        # Verify PipInstall was called with correct packages
        mock_pip_install.assert_called_once_with(packages=cluster.extra_packages)

    @patch("climakitae.util.cluster.PipInstall")
    @patch("climakitae.util.cluster.Gateway")
    @patch("climakitae.util.cluster.GatewayCluster.__init__", return_value=None)
    @patch("climakitae.util.cluster.GatewayCluster.get_client")
    def test_get_client_multiple_clusters(
        self,
        mock_super_get_client: MagicMock,
        mock_init: MagicMock,
        mock_gateway_class: MagicMock,
        mock_pip_install: MagicMock,
    ):
        # Setup
        mock_gateway_instance = MagicMock()
        mock_gateway_class.return_value = mock_gateway_instance

        # Mock PipInstall creation
        mock_pip_install.return_value = MagicMock()

        cluster = Cluster()
        cluster.gateway = mock_gateway_instance

        # Create multiple mock clusters
        cluster1 = MagicMock()
        cluster1.name = "cluster1"
        cluster2 = MagicMock()
        cluster2.name = "cluster2"
        cluster3 = MagicMock()
        cluster3.name = "cluster3"

        cluster.gateway.list_clusters.return_value = [cluster1, cluster2, cluster3]

        mock_connected_cluster = MagicMock()
        mock_client = MagicMock()
        mock_connected_cluster.get_client.return_value = mock_client
        cluster.gateway.connect.return_value = mock_connected_cluster

        # Execute
        result = cluster.get_client()

        # Verify
        assert result == mock_client
        # Should connect to the first cluster (after popping)
        cluster.gateway.connect.assert_called_once_with(
            "cluster3", shutdown_on_close=True
        )
        # Should stop the rest of the clusters
        assert cluster.gateway.stop_cluster.call_count == 2
        cluster.gateway.stop_cluster.assert_has_calls(
            [call("cluster1"), call("cluster2")], any_order=True
        )
        mock_client.register_worker_plugin.assert_called_once()

        # Verify PipInstall was called with correct packages
        mock_pip_install.assert_called_once_with(packages=cluster.extra_packages)

    def test_extra_packages(self):
        # This test verifies the class attribute is set correctly
        assert Cluster.extra_packages == [
            "git+https://github.com/cal-adapt/climakitae.git"
        ]
