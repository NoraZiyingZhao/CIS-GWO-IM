class ArchiveManager:
    def __init__(self, archive_size):
        self.archive = []
        self.archive_size = archive_size

    def update(self, wolves):
        combined = self.archive + wolves
        combined = self._non_dominated_sort(combined)
        if len(combined) > self.archive_size:
            combined = self._reduce_archive(combined)
        self.archive = combined

    def _non_dominated_sort(self, wolves):
        non_dominated = []
        for i, w1 in enumerate(wolves):
            dominated = False
            for j, w2 in enumerate(wolves):
                if i != j and self._dominates(w2, w1):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(w1)
        return non_dominated

    def _dominates(self, a, b):
        return all(x <= y for x, y in zip(a.Cost, b.Cost)) and any(x < y for x, y in zip(a.Cost, b.Cost))

    def _reduce_archive(self, archive):
        return archive[:self.archive_size]  # Simplified truncation

