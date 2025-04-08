from dask_gateway_server.backends.jobqueue.base import JobQueueBackend, JobQueueClusterConfig
from dask_gateway_server.traitlets import Type

from traitlets import Dict, Unicode, default

from enum import Enum
import math, os, pwd, re, shutil

def htcondor_create_execution_script(execution_script, setup_command, execution_command):
    # write script to staging_dir
    with open(execution_script, "w") as f:
        f.write("\n".join([
            "#!/bin/sh",
            "env",
            "ls -l",
            setup_command,
            execution_command
        ]))
        os.chmod(execution_script,  0o755)

def htcondor_create_jdl(cluster_config, execution_script, log_dir, cpus, mem, env, tls_path):
    # ensure log dir is present otherwise condor_submit will fail
    os.makedirs(log_dir, exist_ok=True)

    env["DASK_DISTRIBUTED__COMM__TLS__SCHEDULER__CERT"] = "dask.crt"
    env["DASK_DISTRIBUTED__COMM__TLS__WORKER__CERT"] = "dask.crt"
    env["DASK_DISTRIBUTED__COMM__TLS__SCHEDULER__KEY"] = "dask.pem"
    env["DASK_DISTRIBUTED__COMM__TLS__WORKER__KEY"] = "dask.pem"
    env["DASK_DISTRIBUTED__COMM__TLS__CA_FILE"] = "dask.crt"

    jdl_dict = {"universe": cluster_config.universe,
    "docker_image": cluster_config.docker_image,
    "executable": os.path.relpath(execution_script),
    "docker_network_type": "host",
    "should_transfer_files": "YES",
    "transfer_input_files": ",".join(tls_path),
    "when_to_transfer_output": "ON_EXIT",
    "output": f"{log_dir}/$(cluster).$(process).out",
    "error": f"{log_dir}/$(cluster).$(process).err",
    "log": f"{log_dir}/cluster.log",
    "request_cpus": f"{cpus}",
    "request_memory": f"{mem}",
    "environment": ";".join(f"{key}={value}" for key, value in env.items())
    }
    jdl_dict.update(cluster_config.extra_jdl)

    jdl = "\n".join(f"{key} = {value}" for key, value in jdl_dict.items()) + "\n"
    jdl += "queue 1\n"

    return jdl

def htcondor_memory_format(memory):
    return math.ceil(memory / (1024**2))


class HTCondorJobStates(Enum):
    IDLE = 1
    RUNNING = 2
    REMOVING = 3
    COMPLETED = 4
    HELD = 5
    TRANSFERRING_OUTPUT = 6
    SUSPENDED = 7


class HTCondorClusterConfig(JobQueueClusterConfig):
    """Dask cluster configuration options when running on HTCondor"""
    universe = Unicode("docker", help="The universe to submit jobs to. Defaults to docker", config=True)
    docker_image = Unicode("coffeateam/coffea-dask-cc7-gateway", help="The docker image to run jobs in.", config=True)
    extra_jdl = Dict(help="Additional content of the job description file.", config=True)
    htcondor_staging_directory = Unicode(
    "{home}/.dask-gateway-htcondor/",
    help="""
    The htcondor staging directory for storing files before the job starts.
    A subdirectory will be created for each new cluster which will store
    temporary files for that cluster. On cluster shutdown the subdirectory
    will be removed.
    This field can be a template, which receives the following fields:
    - home (the user's home directory)
    - username (the user's name)
    """,
    config=True,
)


class HTCondorBackend(JobQueueBackend):
    """A backend for deploying Dask on a HTCondor cluster."""

    cluster_config_class = Type(
        "dask_gateway_htcondor.htcondor.HTCondorClusterConfig",
        klass="dask_gateway_server.backends.base.ClusterConfig",
        help="The cluster config class to use",
        config=True,
    )

    @default("submit_command")
    def _default_submit_command(self):
        return shutil.which("condor_submit") or "condor_submit"

    @default("cancel_command")
    def _default_cancel_command(self):
        return shutil.which("condor_rm") or "condor_rm"

    @default("status_command")
    def _default_status_command(self):
        return shutil.which("condor_q") or "condor_q"

    def _get_htcondor_staging_dir(self, cluster):
        htcondor_staging_dir = cluster.config.htcondor_staging_directory.format(
            home=pwd.getpwnam(cluster.username).pw_dir, username=cluster.username
        )
        return os.path.join(htcondor_staging_dir, cluster.name)

    def get_submit_cmd_env_stdin(self, cluster, worker=None):
        cmd = [self.submit_command, "--verbose"]
        htcondor_staging_dir = self._get_htcondor_staging_dir(cluster)

        # ensure that staging_dir exists
        os.makedirs(htcondor_staging_dir, exist_ok=True)

        if worker:
            execution_script = os.path.join(htcondor_staging_dir, f"run_worker_{worker.name}.sh")
            htcondor_create_execution_script(execution_script=execution_script,
                setup_command=cluster.config.worker_setup,
                execution_command=" ".join(self.get_worker_command(cluster, worker.name)))
            env = self.get_worker_env(cluster)
            jdl = htcondor_create_jdl(cluster_config=cluster.config, 
                execution_script=execution_script,
                log_dir=os.path.join(htcondor_staging_dir, f"logs_worker_{worker.name}"),
                cpus=cluster.config.worker_cores, 
                mem=htcondor_memory_format(cluster.config.worker_memory),
                env=env,
                tls_path=self.get_tls_paths(cluster))
        else:
            execution_script = os.path.join(htcondor_staging_dir, f"run_scheduler_{cluster.name}.sh")
            htcondor_create_execution_script(execution_script=execution_script,
                setup_command=cluster.config.scheduler_setup,
                execution_command=" ".join(self.get_scheduler_command(cluster)))
            env = self.get_scheduler_env(cluster)
            jdl = htcondor_create_jdl(cluster_config=cluster.config,
                execution_script=execution_script,
                log_dir=os.path.join(htcondor_staging_dir, f"logs_scheduler_{cluster.name}"),
                cpus=cluster.config.scheduler_cores,
                mem=htcondor_memory_format(cluster.config.scheduler_memory),
                env=env,
                tls_path=self.get_tls_paths(cluster))

        return cmd, env, jdl

    def get_stop_cmd_env(self, job_id):
        return [self.cancel_command, job_id], {}

    def get_status_cmd_env(self, job_ids):
        return [self.status_command, "-af:j", "JobStatus"], {}

    def parse_job_id(self, stdout):
        # search the Job ID in a submit Proc line
        submit_id_pattern = re.compile(r"Proc\s(\d+\.\d+)", flags=re.MULTILINE)
        return submit_id_pattern.search(stdout).group(1)

    def parse_job_states(self, stdout):
        """Checks if job is okay"""
        status_id_pattern = re.compile(r"^([0-9,.]+)\s(\d)$", flags=re.MULTILINE)
        states = {}
        for match in re.finditer(status_id_pattern, stdout):
            job_id, state= match.groups()
            states[job_id] = HTCondorJobStates(int(state)) in (HTCondorJobStates.IDLE, HTCondorJobStates.RUNNING)
        return states
