from omegaconf import DictConfig
from dataclasses import dataclass, asdict

@dataclass
class ConfigData:
    """Class representing configuration data.

    Args:
        setup (bool): Indicates if the setup is enabled or not.
    """

    prompt: str = None
    num_samples: int = None
    image_resolution: int = None
    strength: float = None
    guess_mode: bool = None
    low_threshold: int = None
    high_threshold: int = None
    ddim_steps: int = None
    scale: float = None
    seed: int = None
    eta: float = None
    a_prompt: str = None
    n_prompt: str = None
    mode: str = None

def set_dataclass(self, _dataclass):
    """Sets the dataclass attributes of the object.

    Args:
        _dataclass: An instance of the dataclass.
    Returns:
        None
    """

    data_class_obj = _dataclass()
    for key, value in asdict(data_class_obj).items():
        setattr(self, key, value)

@dataclass
class ConfigData:
    """Class representing configuration data.

    Args:
        setup (bool): Indicates if the setup is enabled or not.
    """

    setup = True


class ConfigSort:
    def __init__(self, config: DictConfig):
        """A class that provides methods for retrieving configuration parameters.

        Args:
            config (DictConfig): The configuration dictionary.
        Attributes:
            config (DictConfig): The configuration dictionary.
        Methods:
            get_config_params(config: DictConfig): Retrieves configuration parameters using the 'consts', 'dt', 'S_IC', and 'iter_max' keys.
            get_ownconfig_params(config): Retrieves configuration parameters using the 'constants', 'dt', 'S_IC', and 'iter_max' keys.
            getconfig_dataclass(config, config_type="default"): Retrieves configuration parameters based on the specified config_type.
        """
        self.config = config

    def get_config_params(self, config: DictConfig):
        """Get the configuration parameters from the given `config` dictionary.

        Args:
            config (DictConfig): The configuration dictionary.
        Returns:
            None
        Raises:
            None
        """

        # self.max_iterations = config.get("iter_max", {}).get("iter_max", "default")

        self.prompt = config.get("prompt", {}).get("prompt")
        self.num_samples = config.get("parameters", {}).get("num_samples")
        self.image_resolution = config.get("parameters", {}).get("image_resolution")
        self.strength = config.get("parameters", {}).get("strength")
        self.guess_mode = config.get("parameters", {}).get("guess_mode")
        self.low_threshold = config.get("parameters", {}).get("low_threshold")
        self.high_threshold = config.get("parameters", {}).get("high_threshold")
        self.ddim_steps = config.get("parameters", {}).get("ddim_steps")
        self.scale = config.get("parameters", {}).get("scale")
        self.seed = config.get("parameters", {}).get("seed")
        self.eta = config.get("parameters", {}).get("eta")
        self.a_prompt = config.get("parameters", {}).get("a_prompt")
        self.n_prompt = config.get("parameters", {}).get("n_prompt")
        self.mode = config.get("mode", {}).get("mode")


    @classmethod
    def getconfig_dataclass(
        cls, config: dataclass
    ) -> dataclass:
        """Retrieves configuration parameters based on the specified config_type.

        Args:
            config: The configuration dictionary.
            config_type (str): The type of configuration "default" or "jupyter".
                               "jupyter" is used for Jupyter notebook configurations.
        Returns:
            ConfigData: An instance of the ConfigData class.
        """

        config_method = cls(config)
        config_method.get_config_params(config)
        return config_method