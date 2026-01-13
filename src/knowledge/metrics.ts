export interface TemplateMetrics {
  selected: number;
  succeeded: number;
  failed: number;
}

export interface GenerationMetrics {
  totalProjects: number;
  duplicateAvoided: number;
  reruns: number;
  tokensUsed: number;
  estimatedCostUsd: number;
}

export class KnowledgeMetrics {
  private templateMetrics = new Map<string, TemplateMetrics>();
  private generation: GenerationMetrics = {
    totalProjects: 0,
    duplicateAvoided: 0,
    reruns: 0,
    tokensUsed: 0,
    estimatedCostUsd: 0
  };

  recordTemplateSelected(templateId: string): void {
    const current = this.templateMetrics.get(templateId) ?? { selected: 0, succeeded: 0, failed: 0 };
    current.selected += 1;
    this.templateMetrics.set(templateId, current);
  }

  recordTemplateOutcome(templateId: string, success: boolean): void {
    const current = this.templateMetrics.get(templateId) ?? { selected: 0, succeeded: 0, failed: 0 };
    if (success) current.succeeded += 1;
    else current.failed += 1;
    this.templateMetrics.set(templateId, current);
  }

  recordProjectIndexed(opts: { isRerun: boolean; tokensUsed?: number; estimatedCostUsd?: number }): void {
    this.generation.totalProjects += 1;
    if (opts.isRerun) this.generation.reruns += 1;
    if (opts.tokensUsed) this.generation.tokensUsed += opts.tokensUsed;
    if (opts.estimatedCostUsd) this.generation.estimatedCostUsd += opts.estimatedCostUsd;
  }

  recordDuplicateAvoided(): void {
    this.generation.duplicateAvoided += 1;
  }

  getTemplateMetrics(): Record<string, TemplateMetrics> {
    return Object.fromEntries(this.templateMetrics.entries());
  }

  getDashboard(): GenerationMetrics & { templates: Record<string, TemplateMetrics> } {
    return {
      ...this.generation,
      templates: this.getTemplateMetrics()
    };
  }
}
