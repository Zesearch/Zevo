"""AWS EC2 GPU provider client.

Third `cloud`-mode backend alongside Vast.ai and Lambda Cloud. The public
surface MIRRORS `LambdaCloudProvider` so the infrastructure agent's cloud block
and the shipped acquire helper drive it with the same calls:

  async with AwsEc2Provider() as p:  # __aenter__/__aexit__ are no-ops
      offers = await p.search_gpus(min_gpu_ram_gb=40, num_gpus=1)
      inst   = await p.create_instance(offer_id=offers[0]["id"], region=offers[0]["region"])
      info   = await p.wait_for_ready(inst["instance_id"], timeout=900)
      ssh    = await p.get_ssh_details(inst["instance_id"])       # user='ubuntu', port 22
      ok     = await p.wait_for_ssh(host=ssh["ssh_host"], key_path=...)
      done   = await p.destroy_instance(inst["instance_id"])

What differs from the marketplace clouds, documented on each method:
  - An offer is an EC2 instance TYPE (``g6e.xlarge``) in the provider's
    region. Only types with a GPU, an x86_64 CPU, an Availability Zone that
    offers them, and a vCPU count within the account's on-demand quota for the
    family are returned, so the agent never picks a type the account cannot
    launch.
  - Price is the on-demand Linux rate from the Pricing API, with a table of
    known rates as the fallback when that API is unreachable. Types with no
    known price are dropped: the cost ledger needs a rate. With
    ``AWS_EC2_MARKET=spot`` the provider rents one-time Spot instances
    instead, checks the Spot quotas, and prices offers from the current Spot
    price history (lowest AZ, which the launch then prefers).
  - A launch needs an AMI, a subnet, a security group that admits SSH, and a
    key pair. All four resolve themselves (Deep Learning Base AMI via SSM, the
    default VPC's subnet in an AZ that offers the type, a ``zevo-gpu-ssh``
    group created on first use, the mounted public key imported under a
    fingerprint name) and each can be pinned with an ``AWS_EC2_*`` variable.
  - Login user is ``ubuntu`` on port 22. Ready status is ``running`` plus a
    public IP; the Deep Learning Base AMI boots with the NVIDIA driver and
    CUDA already installed, so the remote environment helper only adds the
    Python profiles.

Credentials are the standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
pair (boto3 reads them itself); ``ZEVO_AWS_EC2_USE_DEFAULT_CREDENTIALS=1``
accepts whatever the ambient boto3 chain resolves (an instance or task role)
instead. Region is ``AWS_EC2_REGION``, else ``AWS_REGION`` /
``AWS_DEFAULT_REGION``, else ``us-east-1``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from zevo.providers import resolve_ssh_key as _resolve_ssh_key
from zevo.providers.ssh_probe import wait_for_ssh as _wait_for_ssh

logger = logging.getLogger(__name__)

AWS_EC2_SSH_USER = "ubuntu"
AWS_EC2_SSH_PORT = 22
AWS_EC2_CREDENTIAL_ENVS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
DEFAULT_CREDENTIALS_OPT_IN_ENV = "ZEVO_AWS_EC2_USE_DEFAULT_CREDENTIALS"
MAX_INSTANCES_ENV = "ZEVO_AWS_MAX_INSTANCES"

# Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04): driver + CUDA
# preinstalled, bare Python 3.10. Resolved per region through SSM.
DLAMI_SSM_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
SECURITY_GROUP_NAME = "zevo-gpu-ssh"
MANAGED_TAG_KEY = "managed-by"
MANAGED_TAG_VALUE = "zevo"
DEFAULT_ROOT_DISK_GB = 200
MIN_ROOT_DISK_GB = 100

# vCPU quotas per GPU family and market (Service Quotas, service code ec2).
QUOTA_G_AND_VT = "L-DB2E81BA"
QUOTA_P = "L-417A185B"
QUOTA_G_AND_VT_SPOT = "L-3819A6DF"
QUOTA_P_SPOT = "L-7212CCBC"
# `AWS_EC2_MARKET=spot` rents one-time Spot instances (interruption terminates
# them); the default is on-demand. Spot has its own quotas and prices.
MARKET_ENV = "AWS_EC2_MARKET"
MARKETS = ("on-demand", "spot")

# Known on-demand Linux rates (USD/h, us-east-1) used when the Pricing API is
# unavailable to the credentials in use. Pricing API output wins when present.
FALLBACK_USD_PER_HOUR: dict[str, float] = {
    "g4dn.xlarge": 0.526, "g4dn.2xlarge": 0.752, "g4dn.4xlarge": 1.204,
    "g4dn.8xlarge": 2.176, "g4dn.12xlarge": 3.912, "g4dn.16xlarge": 4.352,
    "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g5.8xlarge": 2.448,
    "g5.12xlarge": 5.672, "g5.16xlarge": 4.096, "g5.24xlarge": 8.144, "g5.48xlarge": 16.288,
    "g6.xlarge": 0.805, "g6.2xlarge": 0.978, "g6.4xlarge": 1.323, "g6.8xlarge": 2.014,
    "g6.12xlarge": 4.602, "g6.16xlarge": 3.397, "g6.24xlarge": 6.675, "g6.48xlarge": 13.350,
    "g6e.xlarge": 1.861, "g6e.2xlarge": 2.242, "g6e.4xlarge": 3.004, "g6e.8xlarge": 4.529,
    "g6e.12xlarge": 10.493, "g6e.16xlarge": 7.577, "g6e.24xlarge": 15.066, "g6e.48xlarge": 30.131,
    "gr6.4xlarge": 1.539, "gr6.8xlarge": 2.446,
    "p3.2xlarge": 3.060, "p3.8xlarge": 12.240, "p3.16xlarge": 24.480, "p3dn.24xlarge": 31.212,
    "p4d.24xlarge": 32.773, "p4de.24xlarge": 40.966, "p5.48xlarge": 98.320,
}

_TRUE = ("1", "true", "yes", "on")


def aws_credentials_present(env: Optional[Any] = None) -> bool:
    """Whether this process can drive EC2: both static credential variables,
    or the explicit opt-in to the ambient boto3 credential chain."""
    env = os.environ if env is None else env
    if all(str(env.get(k, "") or "").strip() for k in AWS_EC2_CREDENTIAL_ENVS):
        return True
    return str(env.get(DEFAULT_CREDENTIALS_OPT_IN_ENV, "") or "").strip().lower() in _TRUE


def resolve_region(explicit: Optional[str] = None) -> str:
    for value in (
        explicit,
        os.environ.get("AWS_EC2_REGION"),
        os.environ.get("AWS_REGION"),
        os.environ.get("AWS_DEFAULT_REGION"),
    ):
        if value and str(value).strip():
            return str(value).strip()
    return "us-east-1"


def _default_ssh_key_path() -> str:
    return _resolve_ssh_key("AWS_EC2_SSH_KEY_PATH")


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None) or {}
    return str((response.get("Error") or {}).get("Code") or "")


def mib_to_nameplate_gb(mib: float) -> float:
    """EC2 reports GPU memory in MiB; providers and plans speak in the GB
    printed on the card. EC2 is inconsistent about which figure it stores: the
    usable total for some cards (A10G 22888, L40S 45776, which are 24 GB and
    48 GB decimal) and the nameplate in whole GiB for others (A100 81920,
    T4 16384, which are 80 GB and 16 GB). A whole-GiB figure is read as the
    nameplate; anything else is converted to decimal GB. One rounding, here."""
    mib = float(mib or 0)
    if mib and mib % 1024 == 0:
        return float(mib // 1024)
    return float(round(mib * (2 ** 20) / 1e9))


class AwsEc2Provider:
    """Search, launch, and manage GPU instances on AWS EC2 in one region."""

    def __init__(self, region: Optional[str] = None, session: Any = None, market: Optional[str] = None):
        if session is None and not aws_credentials_present():
            raise FileNotFoundError(
                "No AWS credentials found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY for an IAM identity allowed to run EC2 "
                "instances (ec2:Describe*, ec2:RunInstances, ec2:TerminateInstances, "
                "ec2:CreateTags, ec2:ImportKeyPair, ec2:CreateSecurityGroup, "
                "ec2:AuthorizeSecurityGroupIngress, ssm:GetParameter, "
                "pricing:GetProducts, servicequotas:GetServiceQuota), or set "
                f"{DEFAULT_CREDENTIALS_OPT_IN_ENV}=1 to use the ambient boto3 chain."
            )
        self.region = resolve_region(region)
        self.market = str(market or os.environ.get(MARKET_ENV, "") or "on-demand").strip().lower()
        if self.market not in MARKETS:
            raise ValueError(f"{MARKET_ENV} must be one of {MARKETS}, got {self.market!r}")
        if session is None:
            import boto3

            session = boto3.session.Session(region_name=self.region)
        self._ec2 = session.client("ec2", region_name=self.region)
        self._ssm = session.client("ssm", region_name=self.region)
        self._quotas = session.client("service-quotas", region_name=self.region)
        # The Pricing API is served from a few regions only; us-east-1 answers
        # for every region's prices.
        self._pricing = session.client("pricing", region_name="us-east-1")
        self._type_cache: dict[str, dict] = {}
        self._price_cache: dict[str, float] = {}
        self._quota_cache: dict[str, Optional[float]] = {}
        self._spot_zone: dict[str, str] = {}

    async def __aenter__(self) -> "AwsEc2Provider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    # -----------------------------------------------------------------
    # Catalogue (instance types the account can launch here)
    # -----------------------------------------------------------------

    async def search_gpus(
        self,
        min_gpu_ram_gb: float = 16.0,
        gpu_type: Optional[str] = None,
        num_gpus: int = 1,
        max_dph: Optional[float] = None,
        order: str = "dph_total",  # accepted for VastAIProvider parity; always price-asc
        limit: int = 20,
        min_cpu_ram_gb: float = 0.0,
        min_cpu_cores: int = 0,
    ) -> list[dict]:
        """Return launchable GPU instance types in this region, cheapest first.

        One offer per instance type. A type is launchable when an AZ here
        offers it and its vCPU count fits the family's quota for this market
        (on-demand or Spot); the account's quota is the usual reason an
        otherwise fine type is absent, so the drop is logged with the value.
        Spot offers carry the current lowest Spot price across the offering
        AZs; on-demand offers carry the Pricing API rate.
        """
        return await asyncio.to_thread(
            self._search_gpus, min_gpu_ram_gb, gpu_type, num_gpus, max_dph, limit,
            min_cpu_ram_gb, min_cpu_cores,
        )

    def _search_gpus(
        self, min_gpu_ram_gb, gpu_type, num_gpus, max_dph, limit, min_cpu_ram_gb, min_cpu_cores,
    ) -> list[dict]:
        candidates: list[dict] = []
        for it in self._gpu_instance_types():
            name = it["InstanceType"]
            gpu = (it.get("GpuInfo") or {}).get("Gpus") or [{}]
            gpu = gpu[0]
            n_gpus = int(gpu.get("Count") or 0)
            if n_gpus < 1:
                continue
            if num_gpus and n_gpus != num_gpus:
                continue
            arches = (it.get("ProcessorInfo") or {}).get("SupportedArchitectures") or []
            if "x86_64" not in arches:
                continue
            vram_gb = mib_to_nameplate_gb((gpu.get("MemoryInfo") or {}).get("SizeInMiB") or 0)
            if min_gpu_ram_gb and vram_gb and vram_gb < min_gpu_ram_gb:
                continue
            gpu_name = str(gpu.get("Name") or "")
            if gpu_type and gpu_type.lower().replace("_", "") not in \
                    (name + " " + gpu_name).lower().replace("_", ""):
                continue
            vcpus = int((it.get("VCpuInfo") or {}).get("DefaultVCpus") or 0)
            ram_gb = round(float((it.get("MemoryInfo") or {}).get("SizeInMiB") or 0) / 1024.0, 1)
            if min_cpu_cores and vcpus < min_cpu_cores:
                continue
            if min_cpu_ram_gb and ram_gb < min_cpu_ram_gb:
                continue
            candidates.append({
                "id": name, "gpu_name": gpu_name, "num_gpus": n_gpus, "gpu_ram_gb": float(vram_gb),
                "cpu_cores": vcpus, "ram_gb": ram_gb,
            })
        if not candidates:
            return []

        zones = self._zones_by_type([c["id"] for c in candidates])
        offers: list[dict] = []
        for c in candidates:
            name = c["id"]
            if not zones.get(name):
                continue
            quota_code = self._quota_code_for(name)
            quota = self._quota_value(quota_code) if quota_code else None
            if quota is not None and c["cpu_cores"] > quota:
                logger.info(
                    "AWS EC2: %s needs %d vCPUs but on-demand quota %s is %.0f; dropped",
                    name, c["cpu_cores"], quota_code, quota,
                )
                continue
            dph = self._price(name)
            if dph <= 0:
                logger.warning("AWS EC2: no %s price known for %s; dropped", self.market, name)
                continue
            if max_dph and dph > max_dph:
                continue
            offers.append(_normalize_offer(c, self.region, zones[name], dph, quota, self.market))
        offers.sort(key=lambda o: (o["dph_total"], o["id"]))
        return offers[:limit]

    def _gpu_instance_types(self) -> list[dict]:
        out: list[dict] = []
        token: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {
                "Filters": [{"Name": "instance-type", "Values": ["g*", "p*"]}],
            }
            if token:
                kwargs["NextToken"] = token
            page = self._ec2.describe_instance_types(**kwargs)
            for it in page.get("InstanceTypes") or []:
                if it.get("GpuInfo"):
                    out.append(it)
                    self._type_cache[it["InstanceType"]] = it
            token = page.get("NextToken")
            if not token:
                return out

    def _type_info(self, instance_type: str) -> dict:
        if instance_type not in self._type_cache:
            page = self._ec2.describe_instance_types(InstanceTypes=[instance_type])
            for it in page.get("InstanceTypes") or []:
                self._type_cache[it["InstanceType"]] = it
        return self._type_cache.get(instance_type) or {}

    def _zones_by_type(self, names: list[str]) -> dict[str, list[str]]:
        zones: dict[str, list[str]] = {}
        token: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {
                "LocationType": "availability-zone",
                "Filters": [{"Name": "instance-type", "Values": names}],
            }
            if token:
                kwargs["NextToken"] = token
            page = self._ec2.describe_instance_type_offerings(**kwargs)
            for row in page.get("InstanceTypeOfferings") or []:
                zones.setdefault(row["InstanceType"], []).append(row["Location"])
            token = page.get("NextToken")
            if not token:
                break
        for az_list in zones.values():
            az_list.sort()
        return zones

    def _quota_code_for(self, instance_type: str) -> str:
        family = instance_type.split(".", 1)[0]
        spot = self.market == "spot"
        if family.startswith("p"):
            return QUOTA_P_SPOT if spot else QUOTA_P
        if family.startswith("g"):
            return QUOTA_G_AND_VT_SPOT if spot else QUOTA_G_AND_VT
        return ""

    def _price(self, instance_type: str) -> float:
        return self._spot_price(instance_type) if self.market == "spot" else self._on_demand_price(instance_type)

    def _spot_price(self, instance_type: str) -> float:
        """Lowest current Spot price across this region's AZs (remembering the
        AZ, which the launch then prefers). Falls back to the on-demand rate,
        an upper bound, when the history is empty or unreadable."""
        key = f"spot:{instance_type}"
        if key in self._price_cache:
            return self._price_cache[key]
        price = 0.0
        try:
            from datetime import datetime, timezone

            rows = self._ec2.describe_spot_price_history(
                InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"],
                StartTime=datetime.now(timezone.utc), MaxResults=50,
            ).get("SpotPriceHistory") or []
            best = min(
                ((float(r.get("SpotPrice") or 0), str(r.get("AvailabilityZone") or "")) for r in rows
                 if float(r.get("SpotPrice") or 0) > 0),
                default=(0.0, ""),
            )
            price, zone = round(best[0], 4), best[1]
            if zone:
                self._spot_zone[instance_type] = zone
        except Exception as e:  # noqa: BLE001 - fall back to the on-demand rate
            logger.info("AWS EC2: Spot price history unavailable for %s (%s)", instance_type, e)
        if price <= 0:
            price = self._on_demand_price(instance_type)
        self._price_cache[key] = price
        return price

    def _quota_value(self, code: str) -> Optional[float]:
        if code in self._quota_cache:
            return self._quota_cache[code]
        try:
            value = float(self._quotas.get_service_quota(
                ServiceCode="ec2", QuotaCode=code)["Quota"]["Value"])
        except Exception as e:  # noqa: BLE001 - a missing permission must not hide the catalogue
            logger.warning("AWS EC2: could not read quota %s (%s); not filtering on it", code, e)
            value = None
        self._quota_cache[code] = value
        return value

    def _on_demand_price(self, instance_type: str) -> float:
        if instance_type in self._price_cache:
            return self._price_cache[instance_type]
        price = 0.0
        try:
            resp = self._pricing.get_products(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": self.region},
                    {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                ],
                MaxResults=1,
            )
            for raw in resp.get("PriceList") or []:
                product = json.loads(raw) if isinstance(raw, str) else raw
                for term in (product.get("terms", {}).get("OnDemand") or {}).values():
                    for dim in (term.get("priceDimensions") or {}).values():
                        usd = float((dim.get("pricePerUnit") or {}).get("USD") or 0)
                        if usd > 0:
                            price = round(usd, 4)
        except Exception as e:  # noqa: BLE001 - fall back to the table
            logger.info("AWS EC2: Pricing API unavailable for %s (%s); using known rate", instance_type, e)
        if price <= 0:
            price = float(FALLBACK_USD_PER_HOUR.get(instance_type, 0.0))
        self._price_cache[instance_type] = price
        return price

    # -----------------------------------------------------------------
    # Launch prerequisites (each pinnable through AWS_EC2_*)
    # -----------------------------------------------------------------

    def _ami_id(self) -> str:
        pinned = os.environ.get("AWS_EC2_AMI_ID", "").strip()
        if pinned:
            return pinned
        return str(self._ssm.get_parameter(Name=DLAMI_SSM_PARAMETER)["Parameter"]["Value"])

    def _root_device_name(self, ami_id: str) -> str:
        try:
            images = self._ec2.describe_images(ImageIds=[ami_id]).get("Images") or []
            return str(images[0].get("RootDeviceName") or "/dev/sda1") if images else "/dev/sda1"
        except Exception:  # noqa: BLE001
            return "/dev/sda1"

    def resolve_key_pair_name(self, key_path: str = "") -> str:
        """Return the EC2 key pair holding our public key, importing it if absent.

        `$AWS_EC2_KEY_PAIR_NAME` (trust a pre-created pair) → else read
        `<key_path>.pub`, look up `zevo-<fingerprint>`, → else import it under
        that name. EC2 only installs the key named in RunInstances, so this is
        what makes `ssh -i <key_path> ubuntu@<ip>` work after launch.
        """
        pinned = os.environ.get("AWS_EC2_KEY_PAIR_NAME", "").strip()
        if pinned:
            return pinned
        key_path = key_path or _default_ssh_key_path()
        pub_path = Path(f"{key_path}.pub")
        if not pub_path.exists():
            raise FileNotFoundError(
                f"AWS EC2 launch needs an SSH public key at {pub_path} to import as "
                "a key pair (or set $AWS_EC2_KEY_PAIR_NAME to a pair that already exists)."
            )
        pub = pub_path.read_text(encoding="utf-8").strip()
        body = " ".join(pub.split()[:2])
        name = "zevo-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        try:
            existing = self._ec2.describe_key_pairs(KeyNames=[name]).get("KeyPairs") or []
        except Exception as e:  # noqa: BLE001
            if _error_code(e) != "InvalidKeyPair.NotFound":
                raise
            existing = []
        if existing:
            return name
        self._ec2.import_key_pair(
            KeyName=name, PublicKeyMaterial=body.encode("utf-8"),
            TagSpecifications=[{"ResourceType": "key-pair", "Tags": _managed_tags(name)}],
        )
        logger.info("Imported SSH key as EC2 key pair %r", name)
        return name

    def _subnet_for(self, instance_type: str) -> dict:
        pinned = os.environ.get("AWS_EC2_SUBNET_ID", "").strip()
        if pinned:
            subnets = self._ec2.describe_subnets(SubnetIds=[pinned]).get("Subnets") or []
            if not subnets:
                raise RuntimeError(f"AWS_EC2_SUBNET_ID={pinned} does not exist in {self.region}")
            return subnets[0]
        zones = set(self._zones_by_type([instance_type]).get(instance_type) or [])
        if not zones:
            raise RuntimeError(f"{instance_type} is not offered in any Availability Zone of {self.region}")
        subnets = self._ec2.describe_subnets(
            Filters=[{"Name": "default-for-az", "Values": ["true"]}]).get("Subnets") or []
        preferred = self._spot_zone.get(instance_type, "")
        usable = sorted(
            (s for s in subnets if s.get("AvailabilityZone") in zones),
            key=lambda s: (str(s.get("AvailabilityZone")) != preferred, str(s.get("AvailabilityZone"))),
        )
        if not usable:
            raise RuntimeError(
                f"No default-VPC subnet in {self.region} lies in an AZ that offers {instance_type} "
                f"({sorted(zones)}). Set AWS_EC2_SUBNET_ID to a public subnet in one of those AZs."
            )
        return usable[0]

    def _security_group_id(self, vpc_id: str) -> str:
        pinned = os.environ.get("AWS_EC2_SECURITY_GROUP_ID", "").strip()
        if pinned:
            return pinned
        groups = self._ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [SECURITY_GROUP_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]).get("SecurityGroups") or []
        if groups:
            return str(groups[0]["GroupId"])
        created = self._ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME, VpcId=vpc_id,
            Description="Zevo GPU rentals: SSH from the Zevo backend",
            TagSpecifications=[{"ResourceType": "security-group", "Tags": _managed_tags(SECURITY_GROUP_NAME)}],
        )
        group_id = str(created["GroupId"])
        # The backend's egress address is not fixed (no NAT in front of the
        # scheduler), so SSH is admitted from anywhere; key auth is the gate.
        self._ec2.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": AWS_EC2_SSH_PORT, "ToPort": AWS_EC2_SSH_PORT,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "ssh"}],
        }])
        logger.info("Created security group %s (%s) in %s", SECURITY_GROUP_NAME, group_id, vpc_id)
        return group_id

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def create_instance(
        self,
        offer_id: str,
        region: str = "",
        ssh_key_name: Optional[str] = None,
        ssh_key_path: str = "",
        name: str = "zevo",
        # Accepted for VastAIProvider signature parity. `disk_gb` sizes the
        # root EBS volume; the rest have no EC2 meaning (no docker image:
        # the AMI is the image).
        docker_image: Optional[str] = None,
        disk_gb: int = 0,
        env: Optional[dict[str, str]] = None,
        onstart_cmd: Optional[str] = None,
    ) -> dict:
        """Launch one `offer_id` (an instance type) in this provider's region.
        Returns {instance_id, success}.

        SAFETY CAP mirrors the other clients: refuses to launch when the account
        already holds >= ZEVO_AWS_MAX_INSTANCES (default 1) live Zevo-managed
        instances, so a rerun loop cannot stack boxes and burn money.
        """
        return await asyncio.to_thread(
            self._create_instance, offer_id, region, ssh_key_name, ssh_key_path, name, disk_gb,
        )

    def _create_instance(self, offer_id, region, ssh_key_name, ssh_key_path, name, disk_gb) -> dict:
        if region and region != self.region:
            raise ValueError(
                f"offer region {region!r} does not match this provider's region {self.region!r}; "
                "search and launch through the same provider instance"
            )
        max_active = int(os.environ.get(MAX_INSTANCES_ENV, "1"))
        try:
            existing = self._list_instances()
        except Exception as e:  # noqa: BLE001
            logger.warning("AWS EC2 safety check could not list instances: %s", e)
            existing = []
        active = [i for i in existing if str(i.get("status", "")).lower() not in _GONE_STATES]
        if len(active) >= max_active:
            ids = [i.get("id") for i in active]
            raise RuntimeError(
                f"AWS EC2 safety cap hit: {len(active)} active Zevo instance(s) already "
                f"running {ids}; limit {MAX_INSTANCES_ENV}={max_active}. "
                "Refusing to launch another -- terminate existing instances first."
            )

        ami_id = self._ami_id()
        key_name = ssh_key_name or self.resolve_key_pair_name(ssh_key_path)
        subnet = self._subnet_for(offer_id)
        group_id = self._security_group_id(str(subnet["VpcId"]))
        volume_gb = max(int(disk_gb or 0) or DEFAULT_ROOT_DISK_GB, MIN_ROOT_DISK_GB)
        payload: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": offer_id,
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": key_name,
            "NetworkInterfaces": [{
                "DeviceIndex": 0, "SubnetId": str(subnet["SubnetId"]),
                "Groups": [group_id], "AssociatePublicIpAddress": True,
            }],
            "BlockDeviceMappings": [{
                "DeviceName": self._root_device_name(ami_id),
                "Ebs": {"VolumeSize": volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
            "InstanceInitiatedShutdownBehavior": "terminate",
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": _managed_tags(name)},
                {"ResourceType": "volume", "Tags": _managed_tags(name)},
            ],
        }
        if self.market == "spot":
            payload["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
            }
        resp = self._ec2.run_instances(**payload)
        instances = resp.get("Instances") or []
        if not instances:
            raise RuntimeError(f"No instance in EC2 RunInstances response: {resp}")
        instance_id = str(instances[0]["InstanceId"])
        logger.info("Launched EC2 %s (%s, %s) in %s/%s", instance_id, offer_id, self.market,
                    self.region, subnet.get("AvailabilityZone"))
        return {"instance_id": instance_id, "success": True}

    async def get_instance(self, instance_id: str) -> dict:
        """Fetch one instance, normalized to the dict downstream reads
        (`status` is the EC2 state name; a not-yet-visible instance is
        `not-found`, which the ready poll treats as still pending)."""
        return await asyncio.to_thread(self._get_instance, instance_id)

    def _get_instance(self, instance_id: str) -> dict:
        try:
            reservations = self._ec2.describe_instances(InstanceIds=[instance_id]).get("Reservations") or []
        except Exception as e:  # noqa: BLE001
            if _error_code(e) == "InvalidInstanceID.NotFound":
                return {"id": instance_id, "status": "not-found", "ssh_host": "",
                        "ssh_port": AWS_EC2_SSH_PORT, "ssh_user": AWS_EC2_SSH_USER, "region": self.region}
            raise
        for r in reservations:
            for inst in r.get("Instances") or []:
                return self._normalize_instance(inst)
        return {"id": instance_id, "status": "not-found", "ssh_host": "",
                "ssh_port": AWS_EC2_SSH_PORT, "ssh_user": AWS_EC2_SSH_USER, "region": self.region}

    def _normalize_instance(self, inst: dict) -> dict:
        itype = str(inst.get("InstanceType") or "")
        info = self._type_info(itype) if itype else {}
        gpu = ((info.get("GpuInfo") or {}).get("Gpus") or [{}])[0]
        tags = {t.get("Key"): t.get("Value") for t in inst.get("Tags") or []}
        return {
            "id": str(inst.get("InstanceId") or ""),
            "status": str((inst.get("State") or {}).get("Name") or "unknown"),
            "ssh_host": str(inst.get("PublicIpAddress") or ""),
            "ssh_port": AWS_EC2_SSH_PORT,
            "ssh_user": AWS_EC2_SSH_USER,
            "gpu_name": str(gpu.get("Name") or ""),
            "num_gpus": int(gpu.get("Count") or 0),
            "dph_total": self._price(itype) if itype else 0.0,
            "region": self.region,
            "market": "spot" if inst.get("InstanceLifecycle") == "spot" else "on-demand",
            "zone": str((inst.get("Placement") or {}).get("AvailabilityZone") or ""),
            "instance_type": itype,
            "name": str(tags.get("Name") or ""),
            "launch_time": str(inst.get("LaunchTime") or ""),
        }

    async def list_instances(self) -> list[dict]:
        """Zevo-managed instances in this region (tag managed-by=zevo), as the
        normalized dicts `get_instance` returns. Other EC2 instances in the
        account are not Zevo's to count or destroy."""
        return await asyncio.to_thread(self._list_instances)

    def _list_instances(self) -> list[dict]:
        out: list[dict] = []
        token: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {
                "Filters": [{"Name": f"tag:{MANAGED_TAG_KEY}", "Values": [MANAGED_TAG_VALUE]}],
            }
            if token:
                kwargs["NextToken"] = token
            page = self._ec2.describe_instances(**kwargs)
            for r in page.get("Reservations") or []:
                for inst in r.get("Instances") or []:
                    out.append(self._normalize_instance(inst))
            token = page.get("NextToken")
            if not token:
                return out

    async def destroy_instance(self, instance_id: str) -> bool:
        """Terminate an instance (stops billing; the root volume goes with it).
        Returns True on success or when it is already gone."""
        return await asyncio.to_thread(self._destroy_instance, instance_id)

    def _destroy_instance(self, instance_id: str) -> bool:
        try:
            self._ec2.terminate_instances(InstanceIds=[instance_id])
            logger.info("Terminated EC2 instance %s", instance_id)
            return True
        except Exception as e:  # noqa: BLE001
            if _error_code(e) == "InvalidInstanceID.NotFound":
                return True
            logger.error("Failed to terminate EC2 instance %s: %s", instance_id, e)
            return False

    async def get_ssh_details(self, instance_id: str) -> dict:
        """{ssh_host, ssh_port, ssh_user, instance_id} — user is always 'ubuntu'."""
        info = await self.get_instance(instance_id)
        return {
            "ssh_host": info.get("ssh_host", ""),
            "ssh_port": AWS_EC2_SSH_PORT,
            "ssh_user": AWS_EC2_SSH_USER,
            "instance_id": instance_id,
        }

    async def wait_for_ready(
        self,
        instance_id: str,
        timeout: int = 900,
        poll_interval: int = 15,
    ) -> dict:
        """Poll until state == 'running' AND the instance has a public IP.

        'running' is API-ready, NOT SSH-ready: the caller must additionally
        `wait_for_ssh`. The Deep Learning AMI's first boot takes a few minutes,
        hence the longer default timeout.
        """
        elapsed = 0
        info: dict = {}
        while elapsed < timeout:
            info = await self.get_instance(instance_id)
            status = str(info.get("status", "")).lower()
            if status == "running" and str(info.get("ssh_host") or "").strip():
                return info
            if status in ("shutting-down", "terminated", "stopping", "stopped"):
                raise RuntimeError(
                    f"EC2 instance {instance_id} entered terminal state: {status}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(
            f"EC2 instance {instance_id} not running with a public IP after {timeout}s "
            f"(last status: {info.get('status')}, ip: {info.get('ssh_host') or 'none'})"
        )

    async def wait_for_ssh(
        self,
        *,
        host: str,
        port: int = AWS_EC2_SSH_PORT,
        user: str = AWS_EC2_SSH_USER,
        key_path: str = "",
        timeout: int = 420,
        poll_interval: int = 10,
    ) -> bool:
        """Poll until the box ACCEPTS an SSH connection (see
        `zevo.providers.ssh_probe.wait_for_ssh`). Raises on an empty host."""
        return await _wait_for_ssh(
            host=host, port=port, user=user, key_path=key_path or _default_ssh_key_path(),
            timeout=timeout, poll_interval=poll_interval,
        )


_GONE_STATES = ("shutting-down", "terminated", "not-found")


def _managed_tags(name: str) -> list[dict]:
    return [
        {"Key": "Name", "Value": name},
        {"Key": MANAGED_TAG_KEY, "Value": MANAGED_TAG_VALUE},
    ]


def _normalize_offer(c: dict, region: str, zones: list[str], dph: float, quota: Optional[float], market: str) -> dict:
    """One instance-type offer, shaped like a Vast.ai / Lambda normalized offer
    so the infra agent's cloud block treats every provider uniformly."""
    return {
        "id": c["id"],                          # -> create_instance(offer_id=...)
        "region": region,                       # -> create_instance(region=...)
        "region_desc": ", ".join(zones),
        "zones": list(zones),
        "gpu_name": c["gpu_name"],
        "num_gpus": c["num_gpus"],
        "gpu_ram_gb": c["gpu_ram_gb"],
        "cpu_cores": c["cpu_cores"],
        "ram_gb": c["ram_gb"],
        "disk_gb": float(DEFAULT_ROOT_DISK_GB),  # root EBS volume sized at launch
        "dph_total": dph,
        "reliability": 1.0,                     # first-party hardware; no dud tier
        "verified": True,
        "quota_vcpus": quota,
        "market": market,
        "description": (
            f"{c['id']}: {c['num_gpus']}x {c['gpu_name']} ({c['gpu_ram_gb']:.0f} GB), "
            f"{c['cpu_cores']} vCPU, {c['ram_gb']:.0f} GB RAM, {market}"
        ),
    }
