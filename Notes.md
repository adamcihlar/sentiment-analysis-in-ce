### MetaCentrum
1. Created an [account](https://metavo.metacentrum.cz/osobniv3/wayf/proxy.jsp?locale=en&target=https%3A%2F%2Fsignup.e-infra.cz%2Ffed%2Fregistrar%2F%3Fvo%3Dmeta%26locale%3Den)
2. Filled in application to get access to the resources
3. Followed this [tutorial](https://docs.cloud.muni.cz/cloud/quick-start/) to create my own instance 
4. SSH to the virtual machine
5. Git cloned my repo
6. Installed Docker
7. If you are using Docker on virtual machine always check if mtu is the same or less for docker as for external connection - use command "ip link" ! To have proper internet connection (to install packages, download data, models... anythin) it is needed to create a file /etc/docker/daemon.json with:
{
	"mtu": 1442
}
and restart the docker using: sudo systemctl restart docker
https://mlohr.com/docker-mtu/

### Docker
* Because of wsl I need to start the docker deamon first with: sudo dockerd
