/**
 * Atomic Pipeline - Execution Orchestration
 */

export interface PipelineStage {
  name: string;
  module: string;
  status: 'pending' | 'running' | 'complete' | 'failed';
  progress: number;
  output?: any;
  error?: Error;
  startTime?: Date;
  endTime?: Date;
}

export interface Pipeline {
  id: string;
  stages: PipelineStage[];
  startTime: Date;
  endTime?: Date;
  status: 'initializing' | 'running' | 'complete' | 'failed';
  totalProgress: number;
}

export class PipelineManager {
  private pipelines: Map<string, Pipeline> = new Map();
  
  /**
   * Create new pipeline
   */
  create(stageNames: string[]): Pipeline {
    const pipeline: Pipeline = {
      id: this.generateId(),
      stages: stageNames.map(name => ({
        name,
        module: name,
        status: 'pending',
        progress: 0
      })),
      startTime: new Date(),
      status: 'initializing',
      totalProgress: 0
    };
    
    this.pipelines.set(pipeline.id, pipeline);
    return pipeline;
  }
  
  /**
   * Execute pipeline
   */
  async execute(pipelineId: string, executor: (stage: PipelineStage) => Promise<any>): Promise<void> {
    const pipeline = this.pipelines.get(pipelineId);
    if (!pipeline) throw new Error('Pipeline not found');
    
    pipeline.status = 'running';
    
    try {
      await Promise.all(
        pipeline.stages.map(async (stage) => {
          stage.status = 'running';
          stage.startTime = new Date();
          
          try {
            stage.output = await executor(stage);
            stage.status = 'complete';
            stage.progress = 100;
          } catch (error) {
            stage.status = 'failed';
            stage.error = error as Error;
            throw error;
          } finally {
            stage.endTime = new Date();
          }
        })
      );
      
      pipeline.status = 'complete';
      pipeline.totalProgress = 100;
    } catch (error) {
      pipeline.status = 'failed';
    } finally {
      pipeline.endTime = new Date();
    }
  }
  
  /**
   * Get pipeline status
   */
  getStatus(pipelineId: string): Pipeline | undefined {
    return this.pipelines.get(pipelineId);
  }
  
  /**
   * Update stage progress
   */
  updateProgress(pipelineId: string, stageName: string, progress: number): void {
    const pipeline = this.pipelines.get(pipelineId);
    if (!pipeline) return;
    
    const stage = pipeline.stages.find(s => s.name === stageName);
    if (stage) {
      stage.progress = progress;
      pipeline.totalProgress = pipeline.stages.reduce((sum, s) => sum + s.progress, 0) / pipeline.stages.length;
    }
  }
  
  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `pipe_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const pipelineManager = new PipelineManager();
export default pipelineManager;
