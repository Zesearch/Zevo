"""Unit tests for zevo.providers.aws_ec2.AwsEc2Provider.

AWS is the third `cloud`-mode backend. Its API differs from the marketplace
clouds in ways that would silently break provisioning if a regression crept in:
  - offers are instance TYPES; only those with a GPU, an x86_64 CPU, an AZ that
    offers them, and a vCPU count within the family's on-demand quota may be
    returned (a 0 quota is the usual reason a type is unlaunchable);
  - GPU memory arrives in MiB (A10G: 22888) and must become nameplate GB (24);
  - a launch needs AMI, subnet in an offering AZ, SSH security group and key
    pair, all self-resolving; the ready state is 'running' plus a public IP.

We inject a fake boto3 session so no network is hit, and pin the request
shapes + response normalisation the infra agent and acquire helper rely on.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from zevo.providers.aws_ec2 import provider as mod
from zevo.providers.aws_ec2.provider import (
    AwsEc2Provider, aws_credentials_present, mib_to_nameplate_gb,
    AWS_EC2_SSH_USER, AWS_EC2_SSH_PORT, SECURITY_GROUP_NAME, DLAMI_SSM_PARAMETER,
)


def _type(name, gpu, count, mib, vcpus, mem_mib, arch="x86_64"):
    return {
        "InstanceType": name,
        "VCpuInfo": {"DefaultVCpus": vcpus},
        "MemoryInfo": {"SizeInMiB": mem_mib},
        "ProcessorInfo": {"SupportedArchitectures": [arch]},
        "GpuInfo": {"Gpus": [{"Name": gpu, "Manufacturer": "NVIDIA", "Count": count,
                              "MemoryInfo": {"SizeInMiB": mib}}]},
    }


TYPES = [
    _type("g5.xlarge", "A10G", 1, 22888, 4, 16384),
    _type("g6e.xlarge", "L40S", 1, 45776, 4, 32768),
    _type("g6e.12xlarge", "L40S", 4, 45776, 48, 393216),
    _type("g5g.xlarge", "T4g", 1, 16384, 4, 8192, arch="arm64"),   # ARM -> dropped
    _type("p4d.24xlarge", "A100", 8, 40960, 96, 1179648),
    _type("g4dn.xlarge", "T4", 1, 16384, 4, 16384),                # no AZ offers it -> dropped
    {"InstanceType": "g6f.xlarge", "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 16384},
     "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
     "GpuInfo": {"Gpus": [{"Name": "L4", "Count": 0, "MemoryInfo": {"SizeInMiB": 2861}}]}},  # fractional GPU
]
OFFERINGS = {
    "g5.xlarge": ["us-east-1a", "us-east-1b"],
    "g6e.xlarge": ["us-east-1a", "us-east-1d"],
    "g6e.12xlarge": ["us-east-1a"],
    "p4d.24xlarge": ["us-east-1a"],
    "g5g.xlarge": ["us-east-1a"],
}


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeEc2:
    def __init__(self, *, quotas=None, instances=None, key_pairs=(), groups=()):
        self.instances = instances or {}
        self.key_pairs = set(key_pairs)
        self.groups = list(groups)
        self.calls: list[tuple[str, dict]] = []

    def _rec(self, name, kw):
        self.calls.append((name, kw))

    def describe_instance_types(self, **kw):
        self._rec("describe_instance_types", kw)
        if "InstanceTypes" in kw:
            return {"InstanceTypes": [t for t in TYPES if t["InstanceType"] in kw["InstanceTypes"]]}
        return {"InstanceTypes": TYPES}

    def describe_instance_type_offerings(self, **kw):
        self._rec("describe_instance_type_offerings", kw)
        names = kw["Filters"][0]["Values"]
        rows = [{"InstanceType": n, "Location": az} for n in names for az in OFFERINGS.get(n, [])]
        return {"InstanceTypeOfferings": rows}

    def describe_subnets(self, **kw):
        self._rec("describe_subnets", kw)
        if "SubnetIds" in kw:
            return {"Subnets": [{"SubnetId": kw["SubnetIds"][0], "VpcId": "vpc-pinned", "AvailabilityZone": "us-east-1z"}]}
        return {"Subnets": [
            {"SubnetId": "subnet-e", "VpcId": "vpc-default", "AvailabilityZone": "us-east-1e"},
            {"SubnetId": "subnet-d", "VpcId": "vpc-default", "AvailabilityZone": "us-east-1d"},
            {"SubnetId": "subnet-a", "VpcId": "vpc-default", "AvailabilityZone": "us-east-1a"},
        ]}

    def describe_security_groups(self, **kw):
        self._rec("describe_security_groups", kw)
        return {"SecurityGroups": [{"GroupId": g} for g in self.groups]}

    def create_security_group(self, **kw):
        self._rec("create_security_group", kw)
        self.groups.append("sg-new")
        return {"GroupId": "sg-new"}

    def authorize_security_group_ingress(self, **kw):
        self._rec("authorize_security_group_ingress", kw)
        return {}

    def describe_key_pairs(self, **kw):
        self._rec("describe_key_pairs", kw)
        name = kw["KeyNames"][0]
        if name not in self.key_pairs:
            raise ClientError("InvalidKeyPair.NotFound")
        return {"KeyPairs": [{"KeyName": name}]}

    def import_key_pair(self, **kw):
        self._rec("import_key_pair", kw)
        self.key_pairs.add(kw["KeyName"])
        return {"KeyName": kw["KeyName"]}

    def describe_images(self, **kw):
        self._rec("describe_images", kw)
        return {"Images": [{"ImageId": kw["ImageIds"][0], "RootDeviceName": "/dev/sda1"}]}

    def run_instances(self, **kw):
        self._rec("run_instances", kw)
        inst = {"InstanceId": "i-new", "InstanceType": kw["InstanceType"], "State": {"Name": "pending"},
                "Tags": kw["TagSpecifications"][0]["Tags"], "Placement": {"AvailabilityZone": "us-east-1a"}}
        self.instances["i-new"] = inst
        return {"Instances": [inst]}

    def describe_instances(self, **kw):
        self._rec("describe_instances", kw)
        if "InstanceIds" in kw:
            iid = kw["InstanceIds"][0]
            if iid not in self.instances:
                raise ClientError("InvalidInstanceID.NotFound")
            return {"Reservations": [{"Instances": [self.instances[iid]]}]}
        managed = [i for i in self.instances.values()
                   if {"Key": "managed-by", "Value": "zevo"} in (i.get("Tags") or [])]
        return {"Reservations": [{"Instances": managed}]}

    def terminate_instances(self, **kw):
        self._rec("terminate_instances", kw)
        for iid in kw["InstanceIds"]:
            if iid not in self.instances:
                raise ClientError("InvalidInstanceID.NotFound")
            self.instances[iid]["State"] = {"Name": "shutting-down"}
        return {}


class FakeSsm:
    def get_parameter(self, **kw):
        assert kw["Name"] == DLAMI_SSM_PARAMETER
        return {"Parameter": {"Value": "ami-dlami"}}


class FakeQuotas:
    def __init__(self, values):
        self.values = values

    def get_service_quota(self, **kw):
        if kw["QuotaCode"] not in self.values:
            raise ClientError("AccessDeniedException")
        return {"Quota": {"Value": self.values[kw["QuotaCode"]]}}


class FakePricing:
    def __init__(self, prices=None, fail=False):
        self.prices = prices or {}
        self.fail = fail

    def get_products(self, **kw):
        if self.fail:
            raise ClientError("AccessDeniedException")
        itype = next(f["Value"] for f in kw["Filters"] if f["Field"] == "instanceType")
        if itype not in self.prices:
            return {"PriceList": []}
        return {"PriceList": [json.dumps({"terms": {"OnDemand": {"x": {"priceDimensions": {
            "y": {"pricePerUnit": {"USD": str(self.prices[itype])}}}}}}})]}


class FakeSession:
    def __init__(self, ec2, quotas=None, pricing=None):
        self.clients = {"ec2": ec2, "ssm": FakeSsm(),
                        "service-quotas": quotas or FakeQuotas({mod.QUOTA_G_AND_VT: 64, mod.QUOTA_P: 0}),
                        "pricing": pricing or FakePricing({"g5.xlarge": 1.006, "g6e.xlarge": 1.861,
                                                           "g6e.12xlarge": 10.493, "p4d.24xlarge": 32.77})}

    def client(self, name, region_name=None):
        return self.clients[name]


def _provider(monkeypatch, ec2=None, **kw):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    for var in ("AWS_EC2_SUBNET_ID", "AWS_EC2_SECURITY_GROUP_ID", "AWS_EC2_KEY_PAIR_NAME",
                "AWS_EC2_AMI_ID", "AWS_EC2_REGION", "ZEVO_AWS_MAX_INSTANCES"):
        monkeypatch.delenv(var, raising=False)
    ec2 = ec2 or FakeEc2()
    return AwsEc2Provider(session=FakeSession(ec2, **kw)), ec2


def _key(tmp_path, monkeypatch):
    key = tmp_path / "id_ed25519"
    key.write_text("private")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAATESTKEY comment@host\n")
    monkeypatch.setenv("AWS_EC2_SSH_KEY_PATH", str(key))
    return str(key)


# ----------------------------------------------------------------- credentials

def test_credentials_present_needs_both_static_keys_or_the_opt_in():
    assert not aws_credentials_present({"AWS_ACCESS_KEY_ID": "AKIA" + "A" * 16})
    assert aws_credentials_present({"AWS_ACCESS_KEY_ID": "AKIA" + "A" * 16, "AWS_SECRET_ACCESS_KEY": "s" * 40})
    assert aws_credentials_present({"ZEVO_AWS_EC2_USE_DEFAULT_CREDENTIALS": "1"})
    assert not aws_credentials_present({})


def test_constructor_refuses_without_credentials(monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "ZEVO_AWS_EC2_USE_DEFAULT_CREDENTIALS"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FileNotFoundError, match="AWS_ACCESS_KEY_ID"):
        AwsEc2Provider()


def test_region_resolution_prefers_ec2_specific_variable(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_EC2_REGION", "eu-west-1")
    assert mod.resolve_region() == "eu-west-1"
    monkeypatch.delenv("AWS_EC2_REGION")
    assert mod.resolve_region() == "us-west-2"


# ---------------------------------------------------------------------- search

def test_mib_becomes_nameplate_gb():
    assert mib_to_nameplate_gb(22888) == 24
    assert mib_to_nameplate_gb(45776) == 48
    assert mib_to_nameplate_gb(81920) == 80   # A100/H100 80GB: EC2 stores whole GiB
    assert mib_to_nameplate_gb(16384) == 16   # T4
    assert mib_to_nameplate_gb(98304) == 96   # RTX PRO 6000
    assert mib_to_nameplate_gb(0) == 0


def test_search_returns_launchable_types_cheapest_first(monkeypatch):
    p, ec2 = _provider(monkeypatch)
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=16, num_gpus=1))
    ids = [o["id"] for o in offers]
    # g4dn: no AZ; g5g: arm64; g6f: fractional GPU; p4d/g6e.12x: num_gpus != 1
    assert ids == ["g5.xlarge", "g6e.xlarge"]
    a10g = offers[0]
    assert a10g["gpu_name"] == "A10G" and a10g["gpu_ram_gb"] == 24.0 and a10g["dph_total"] == 1.006
    assert a10g["region"] == "us-east-1" and a10g["zones"] == ["us-east-1a", "us-east-1b"]
    assert a10g["cpu_cores"] == 4 and a10g["ram_gb"] == 16.0 and a10g["quota_vcpus"] == 64
    assert a10g["verified"] is True and a10g["reliability"] == 1.0


def test_search_drops_types_over_the_family_quota(monkeypatch):
    p, _ = _provider(monkeypatch, quotas=FakeQuotas({mod.QUOTA_G_AND_VT: 8, mod.QUOTA_P: 0}))
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=0, num_gpus=0))
    ids = [o["id"] for o in offers]
    assert "g6e.12xlarge" not in ids and "p4d.24xlarge" not in ids   # 48 and 96 vCPUs over quota
    assert ids == ["g5.xlarge", "g6e.xlarge"]


def test_search_with_zero_quota_returns_nothing(monkeypatch):
    p, _ = _provider(monkeypatch, quotas=FakeQuotas({mod.QUOTA_G_AND_VT: 0, mod.QUOTA_P: 0}))
    assert asyncio.run(p.search_gpus(min_gpu_ram_gb=16)) == []


def test_unreadable_quota_does_not_hide_the_catalogue(monkeypatch):
    p, _ = _provider(monkeypatch, quotas=FakeQuotas({}))
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=16, num_gpus=1))
    assert [o["id"] for o in offers] == ["g5.xlarge", "g6e.xlarge"]
    assert offers[0]["quota_vcpus"] is None


def test_search_filters_vram_type_and_price(monkeypatch):
    p, _ = _provider(monkeypatch)
    assert [o["id"] for o in asyncio.run(p.search_gpus(min_gpu_ram_gb=40))] == ["g6e.xlarge"]
    assert [o["id"] for o in asyncio.run(p.search_gpus(min_gpu_ram_gb=16, gpu_type="a10g"))] == ["g5.xlarge"]
    assert [o["id"] for o in asyncio.run(p.search_gpus(min_gpu_ram_gb=16, max_dph=1.5))] == ["g5.xlarge"]
    assert [o["id"] for o in asyncio.run(p.search_gpus(min_gpu_ram_gb=16, min_cpu_ram_gb=20))] == ["g6e.xlarge"]


def test_pricing_api_failure_falls_back_to_known_rates(monkeypatch):
    p, _ = _provider(monkeypatch, pricing=FakePricing(fail=True))
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=16, num_gpus=1))
    by_id = {o["id"]: o["dph_total"] for o in offers}
    assert by_id == {"g5.xlarge": mod.FALLBACK_USD_PER_HOUR["g5.xlarge"],
                     "g6e.xlarge": mod.FALLBACK_USD_PER_HOUR["g6e.xlarge"]}


# ---------------------------------------------------------------------- launch

def test_create_resolves_ami_subnet_group_and_key_then_launches(tmp_path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    p, ec2 = _provider(monkeypatch)
    out = asyncio.run(p.create_instance("g6e.xlarge", "us-east-1", ssh_key_path=key, name="zevo-abc", disk_gb=150))
    assert out == {"instance_id": "i-new", "success": True}

    names = [c[0] for c in ec2.calls]
    assert "import_key_pair" in names and "create_security_group" in names
    run = dict(ec2.calls)["run_instances"]
    assert run["ImageId"] == "ami-dlami" and run["InstanceType"] == "g6e.xlarge"
    assert run["MinCount"] == run["MaxCount"] == 1
    assert run["KeyName"].startswith("zevo-")
    nic = run["NetworkInterfaces"][0]
    # the default-VPC subnet must sit in an AZ that offers g6e.xlarge (a or d), not e
    assert nic["SubnetId"] == "subnet-a" and nic["Groups"] == ["sg-new"] and nic["AssociatePublicIpAddress"] is True
    ebs = run["BlockDeviceMappings"][0]
    assert ebs["DeviceName"] == "/dev/sda1" and ebs["Ebs"] == {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True}
    assert run["InstanceInitiatedShutdownBehavior"] == "terminate"
    tags = run["TagSpecifications"][0]["Tags"]
    assert {"Key": "Name", "Value": "zevo-abc"} in tags and {"Key": "managed-by", "Value": "zevo"} in tags

    ingress = dict(ec2.calls)["authorize_security_group_ingress"]
    perm = ingress["IpPermissions"][0]
    assert perm["FromPort"] == perm["ToPort"] == 22 and perm["IpRanges"][0]["CidrIp"] == "0.0.0.0/0"
    sg = dict(ec2.calls)["create_security_group"]
    assert sg["GroupName"] == SECURITY_GROUP_NAME and sg["VpcId"] == "vpc-default"


def test_create_reuses_existing_key_pair_and_group_and_enforces_min_disk(tmp_path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    p, ec2 = _provider(monkeypatch, ec2=FakeEc2(groups=["sg-existing"]))
    first = asyncio.run(p.create_instance("g5.xlarge", "", ssh_key_path=key))
    assert first["success"]
    key_name = dict(ec2.calls)["run_instances"]["KeyName"]
    assert key_name in ec2.key_pairs
    assert dict(ec2.calls)["run_instances"]["NetworkInterfaces"][0]["Groups"] == ["sg-existing"]
    assert "create_security_group" not in [c[0] for c in ec2.calls]
    assert dict(ec2.calls)["run_instances"]["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == mod.DEFAULT_ROOT_DISK_GB

    ec2.calls.clear(); ec2.instances.clear()
    asyncio.run(p.create_instance("g5.xlarge", "", ssh_key_path=key, disk_gb=20))
    assert "import_key_pair" not in [c[0] for c in ec2.calls]
    assert dict(ec2.calls)["run_instances"]["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == mod.MIN_ROOT_DISK_GB


def test_create_honours_pins(tmp_path, monkeypatch):
    _key(tmp_path, monkeypatch)
    p, ec2 = _provider(monkeypatch)
    monkeypatch.setenv("AWS_EC2_AMI_ID", "ami-pinned")
    monkeypatch.setenv("AWS_EC2_SUBNET_ID", "subnet-pinned")
    monkeypatch.setenv("AWS_EC2_SECURITY_GROUP_ID", "sg-pinned")
    monkeypatch.setenv("AWS_EC2_KEY_PAIR_NAME", "my-pair")
    asyncio.run(p.create_instance("g5.xlarge"))
    run = dict(ec2.calls)["run_instances"]
    assert run["ImageId"] == "ami-pinned" and run["KeyName"] == "my-pair"
    assert run["NetworkInterfaces"][0] == {"DeviceIndex": 0, "SubnetId": "subnet-pinned",
                                           "Groups": ["sg-pinned"], "AssociatePublicIpAddress": True}
    assert not any(c[0] in ("import_key_pair", "create_security_group", "describe_instance_type_offerings")
                   for c in ec2.calls)


def test_create_rejects_a_foreign_region_and_the_safety_cap(tmp_path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    live = {"i-live": {"InstanceId": "i-live", "InstanceType": "g5.xlarge", "State": {"Name": "running"},
                       "Tags": [{"Key": "managed-by", "Value": "zevo"}]}}
    p, _ = _provider(monkeypatch, ec2=FakeEc2(instances=live))
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(p.create_instance("g5.xlarge", "eu-west-1", ssh_key_path=key))
    with pytest.raises(RuntimeError, match="safety cap"):
        asyncio.run(p.create_instance("g5.xlarge", "us-east-1", ssh_key_path=key))
    monkeypatch.setenv("ZEVO_AWS_MAX_INSTANCES", "2")
    assert asyncio.run(p.create_instance("g5.xlarge", "us-east-1", ssh_key_path=key))["success"]


def test_create_fails_clearly_when_no_default_subnet_offers_the_type(tmp_path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    p, _ = _provider(monkeypatch)
    OFFERINGS["g6e.12xlarge"] = ["us-east-1c"]   # no default subnet in 1c
    try:
        with pytest.raises(RuntimeError, match="AWS_EC2_SUBNET_ID"):
            asyncio.run(p.create_instance("g6e.12xlarge", "", ssh_key_path=key))
    finally:
        OFFERINGS["g6e.12xlarge"] = ["us-east-1a"]


# ------------------------------------------------------------ instance state

def test_get_instance_normalizes_and_ready_waits_for_a_public_ip(monkeypatch):
    inst = {"InstanceId": "i-1", "InstanceType": "g6e.xlarge", "State": {"Name": "pending"},
            "Tags": [{"Key": "Name", "Value": "zevo-x"}, {"Key": "managed-by", "Value": "zevo"}],
            "Placement": {"AvailabilityZone": "us-east-1a"}}
    p, ec2 = _provider(monkeypatch, ec2=FakeEc2(instances={"i-1": inst}))
    sleeps: list[int] = []

    async def fake_sleep(n):
        sleeps.append(n)
        if len(sleeps) == 1:
            inst["State"] = {"Name": "running"}          # running, still no IP -> keep polling
        elif len(sleeps) == 2:
            inst["PublicIpAddress"] = "203.0.113.9"
    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    info = asyncio.run(p.wait_for_ready("i-1", timeout=100, poll_interval=10))
    assert len(sleeps) == 2
    assert info["status"] == "running" and info["ssh_host"] == "203.0.113.9"
    assert info["ssh_user"] == AWS_EC2_SSH_USER and info["ssh_port"] == AWS_EC2_SSH_PORT
    assert info["gpu_name"] == "L40S" and info["num_gpus"] == 1 and info["dph_total"] == 1.861
    assert info["region"] == "us-east-1" and info["zone"] == "us-east-1a" and info["name"] == "zevo-x"

    details = asyncio.run(p.get_ssh_details("i-1"))
    assert details == {"ssh_host": "203.0.113.9", "ssh_port": 22, "ssh_user": "ubuntu", "instance_id": "i-1"}


def test_ready_treats_not_found_as_pending_and_terminated_as_fatal(monkeypatch):
    inst = {"InstanceId": "i-2", "InstanceType": "g5.xlarge", "State": {"Name": "pending"}, "Tags": []}
    ec2 = FakeEc2()
    p, _ = _provider(monkeypatch, ec2=ec2)
    sleeps: list[int] = []

    async def fake_sleep(n):
        sleeps.append(n)
        if len(sleeps) == 1:
            ec2.instances["i-2"] = inst                   # becomes visible after eventual consistency
        else:
            inst["State"] = {"Name": "terminated"}
    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    assert asyncio.run(p.get_instance("i-2"))["status"] == "not-found"
    with pytest.raises(RuntimeError, match="terminal state: terminated"):
        asyncio.run(p.wait_for_ready("i-2", timeout=100, poll_interval=10))


def test_list_and_destroy_only_touch_managed_instances(monkeypatch):
    instances = {
        "i-zevo": {"InstanceId": "i-zevo", "InstanceType": "g5.xlarge", "State": {"Name": "running"},
                   "Tags": [{"Key": "managed-by", "Value": "zevo"}], "PublicIpAddress": "203.0.113.1"},
        "i-other": {"InstanceId": "i-other", "InstanceType": "m5.large", "State": {"Name": "running"}, "Tags": []},
    }
    p, ec2 = _provider(monkeypatch, ec2=FakeEc2(instances=instances))
    listed = asyncio.run(p.list_instances())
    assert [i["id"] for i in listed] == ["i-zevo"]
    assert listed[0]["status"] == "running" and listed[0]["ssh_host"] == "203.0.113.1"

    assert asyncio.run(p.destroy_instance("i-zevo")) is True
    assert instances["i-zevo"]["State"]["Name"] == "shutting-down"
    assert asyncio.run(p.destroy_instance("i-gone")) is True   # already gone counts as destroyed
    assert dict(ec2.calls)["describe_instances"]["Filters"] == [{"Name": "tag:managed-by", "Values": ["zevo"]}]


def test_wait_for_ssh_refuses_an_empty_host(monkeypatch):
    p, _ = _provider(monkeypatch)
    with pytest.raises(ValueError, match="requires a host"):
        asyncio.run(p.wait_for_ssh(host="", key_path="/k"))


# ------------------------------------------------------------------------ spot

class SpotEc2(FakeEc2):
    def __init__(self, history=None, **kw):
        super().__init__(**kw)
        self.history = history if history is not None else [
            {"InstanceType": "g6e.xlarge", "AvailabilityZone": "us-east-1a", "SpotPrice": "0.91"},
            {"InstanceType": "g6e.xlarge", "AvailabilityZone": "us-east-1d", "SpotPrice": "0.74"},
            {"InstanceType": "g5.xlarge", "AvailabilityZone": "us-east-1a", "SpotPrice": "0.45"},
        ]

    def describe_spot_price_history(self, **kw):
        self._rec("describe_spot_price_history", kw)
        return {"SpotPriceHistory": [r for r in self.history if r["InstanceType"] in kw["InstanceTypes"]]}


def test_market_variable_is_validated(monkeypatch):
    monkeypatch.setenv("AWS_EC2_MARKET", "preemptible")
    with pytest.raises(ValueError, match="AWS_EC2_MARKET"):
        AwsEc2Provider(session=FakeSession(FakeEc2()))


def test_spot_search_uses_spot_quotas_and_spot_prices(monkeypatch):
    monkeypatch.setenv("AWS_EC2_MARKET", "spot")
    quotas = FakeQuotas({mod.QUOTA_G_AND_VT: 0, mod.QUOTA_G_AND_VT_SPOT: 64, mod.QUOTA_P_SPOT: 0})
    p, ec2 = _provider(monkeypatch, ec2=SpotEc2(), quotas=quotas)
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=16, num_gpus=1))
    assert [(o["id"], o["dph_total"], o["market"]) for o in offers] == [
        ("g5.xlarge", 0.45, "spot"), ("g6e.xlarge", 0.74, "spot")]
    assert offers[1]["quota_vcpus"] == 64
    assert "describe_spot_price_history" in [c[0] for c in ec2.calls]


def test_spot_price_falls_back_to_on_demand_when_history_is_empty(monkeypatch):
    monkeypatch.setenv("AWS_EC2_MARKET", "spot")
    quotas = FakeQuotas({mod.QUOTA_G_AND_VT_SPOT: 64, mod.QUOTA_P_SPOT: 0})
    p, _ = _provider(monkeypatch, ec2=SpotEc2(history=[]), quotas=quotas)
    offers = asyncio.run(p.search_gpus(min_gpu_ram_gb=40, num_gpus=1))
    assert [(o["id"], o["dph_total"]) for o in offers] == [("g6e.xlarge", 1.861)]


def test_spot_launch_sets_market_options_and_prefers_the_cheapest_zone(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_EC2_MARKET", "spot")
    key = _key(tmp_path, monkeypatch)
    quotas = FakeQuotas({mod.QUOTA_G_AND_VT_SPOT: 64, mod.QUOTA_P_SPOT: 0})
    p, ec2 = _provider(monkeypatch, ec2=SpotEc2(), quotas=quotas)
    asyncio.run(p.search_gpus(min_gpu_ram_gb=40, num_gpus=1))   # learns us-east-1d is cheapest for g6e.xlarge
    asyncio.run(p.create_instance("g6e.xlarge", "us-east-1", ssh_key_path=key))
    run = dict(ec2.calls)["run_instances"]
    assert run["InstanceMarketOptions"] == {
        "MarketType": "spot",
        "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
    }
    assert run["NetworkInterfaces"][0]["SubnetId"] == "subnet-d"


def test_on_demand_launch_has_no_market_options(tmp_path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    p, ec2 = _provider(monkeypatch)
    asyncio.run(p.create_instance("g5.xlarge", "", ssh_key_path=key))
    assert "InstanceMarketOptions" not in dict(ec2.calls)["run_instances"]
