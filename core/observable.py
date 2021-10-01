from abc import ABC, abstractmethod


class Observable(ABC):
    """
    The Observable interface declares a set of methods for managing subscribers.
    This abstract class is used to help propagate changes throughout a dataset.
    """

    @abstractmethod
    def attach(self, observer):
        """
        Attach an observer to the subject.
        """
        pass

    @abstractmethod
    def detach(self, observer):
        """
        Detach an observer from the subject.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all observers about an event.
        """
        pass