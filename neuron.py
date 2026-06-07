import torch

class Neuron:
    """
    EFCT нейрон v3 — правильная COM метрика.
    
    Ключевой инсайт:
    Confidence модели с random весами = ~1/32000 = 0.00003
    Это не "плохо" — это baseline для данного узла.
    
    COM должна измерять ОТНОСИТЕЛЬНОЕ улучшение от baseline,
    а не абсолютное значение сигнала.
    
    Fisher-Rao метрика: T_subj = ∫ds
    ds = изменение уверенности относительно среднего
    """
    
    def __init__(self, size=512, neuron_id="unnamed"):
        self.id = neuron_id
        self.size = size
        self.gates = torch.zeros(size)
        self.phase = torch.zeros(size)
        self.com_history = []
        self.tasks_done = 0
        
        # Скользящее среднее для baseline
        self.baseline = 0.5
        self.baseline_alpha = 0.1  # скорость адаптации baseline
    
    def forward(self, activations):
        rotated = activations * torch.cos(self.phase)
        output = rotated * (1.0 + self.gates)
        return output
    
    def local_update(self, raw_quality):
        """
        COM v3: обновляем relative to baseline.
        
        raw_quality = любое число (confidence, cosine sim, etc.)
        signal = (raw_quality - baseline) / baseline  → нормализован
        """
        with torch.no_grad():
            # Нормализованный сигнал: +1 = вдвое лучше baseline
            if self.baseline > 1e-10:
                signal = (raw_quality - self.baseline) / (abs(self.baseline) + 1e-10)
            else:
                signal = 0.0
            
            # Testing Effect: лучше baseline → усиливаем gates
            if signal > 0:
                beta = 0.05
                self.gates += beta * signal * (1.0 - self.gates.abs().clamp(max=0.5))
                self.phase += 0.01 * signal
            # Zeno Effect: хуже baseline → слабо тянем к нулю
            elif signal < -0.1:
                self.gates *= (1.0 - 0.001 * abs(signal))
            
            # Обновляем baseline (экспоненциальное скользящее среднее)
            self.baseline = (1 - self.baseline_alpha) * self.baseline + \
                           self.baseline_alpha * raw_quality
        
        self.com_history.append(round(signal, 6))
        self.tasks_done += 1
    
    def identity_distance(self):
        return float(self.gates.abs().mean() + self.phase.abs().mean())

    def save(self, path: str):
        torch.save({
            "gates":    self.gates,
            "phase":    self.phase,
            "baseline": self.baseline,
            "tasks":    self.tasks_done,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Neuron":
        obj  = cls(**kwargs)
        data = torch.load(path, weights_only=True)
        obj.gates      = data["gates"]
        obj.phase      = data.get("phase",    torch.zeros_like(obj.gates))
        obj.baseline   = data.get("baseline", None)
        obj.tasks_done = data.get("tasks",    0)
        return obj

