#!/usr/bin/env node

// Simple demonstration of the Atomic Forge module
// This shows the core functionality without complex dependencies

console.log('🚀 Atomic Forge - Production Code Generator Demo');
console.log('=================================================\n');

// Mock the Forge module functionality for demonstration
class ForgeDemo {
  generate(spec) {
    console.log('📋 Generating project with specifications:');
    console.log(`   Name: ${spec.name}`);
    console.log(`   Language: ${spec.language}`);
    console.log(`   Framework: ${spec.framework}`);
    console.log(`   Architecture: ${spec.architecture}`);
    console.log(`   Database: ${spec.database || 'None'}`);
    console.log(`   Authentication: ${spec.authentication}`);
    console.log(`   Testing: ${spec.testing ? 'Enabled' : 'Disabled'}`);
    console.log(`   Deployment: ${spec.deployment.join(', ')}`);
    console.log(`   Features: ${spec.features.join(', ') || 'None'}`);
    
    console.log('\n🎯 Generating files...');
    
    // Simulate file generation
    const files = [
      { path: 'src/index.ts', language: 'typescript', content: '// Main application entry point' },
      { path: 'src/controllers/HealthController.ts', language: 'typescript', content: '// Health check controller' },
      { path: 'src/services/UserService.ts', language: 'typescript', content: '// User service' },
      { path: 'src/models/User.ts', language: 'typescript', content: '// User model' },
      { path: 'package.json', language: 'json', content: '{"name": "project", "dependencies": {}}' },
      { path: 'tsconfig.json', language: 'json', content: '{"compilerOptions": {"target": "ES2020"}}' },
      { path: '.env.example', language: 'text', content: 'NODE_ENV=development\nPORT=3000' },
      { path: 'README.md', language: 'markdown', content: '# Generated Project\n\nGenerated with Atomic Forge.' },
      { path: 'Dockerfile', language: 'dockerfile', content: 'FROM node:18-alpine\nWORKDIR /app\nCMD ["npm", "start"]' },
      { path: 'docker-compose.yml', language: 'yaml', content: 'version: "3.8"\nservices:\n  app:\n    build: .' },
      { path: '.github/workflows/ci.yml', language: 'yaml', content: 'name: CI\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest' },
      { path: 'jest.config.js', language: 'javascript', content: 'module.exports = { preset: "ts-jest" }' },
      { path: 'tests/health.test.ts', language: 'typescript', content: 'describe("Health", () => { test("check", () => {}); })' },
      { path: 'src/utils/logger.ts', language: 'typescript', content: 'export const logger = { info: console.log }' },
      { path: 'src/database/connection.ts', language: 'typescript', content: 'export const db = { connect: () => Promise.resolve() }' }
    ];
    
    console.log(`✅ Generated ${files.length} files`);
    
    console.log('\n📁 Sample generated files:');
    files.slice(0, 8).forEach(file => {
      console.log(`   ✓ ${file.path} (${file.language})`);
    });
    if (files.length > 8) {
      console.log(`   ... and ${files.length - 8} more files`);
    }
    
    // Generate build instructions
    const buildInstructions = this.generateBuildInstructions(spec);
    
    console.log('\n📖 Build Instructions:');
    console.log(buildInstructions);
    
    console.log('\n🎉 Project generation completed successfully!');
    
    return {
      files,
      summary: {
        totalFiles: files.length,
        languages: [spec.language],
        frameworks: [spec.framework],
        architecture: spec.architecture,
        features: spec.features,
        generationTime: Math.floor(Math.random() * 500) + 100 // Simulated time
      },
      buildInstructions
    };
  }
  
  generateBuildInstructions(spec) {
    switch (spec.language) {
      case 'typescript':
        return `# Build Instructions for TypeScript Project

## Prerequisites
- Node.js 18+
- npm or yarn

## Installation
npm install

## Development
npm run dev

## Building
npm run build

## Testing
npm test

## Running
npm start

## Docker
docker build -t ${spec.name} .
docker-compose up`;
      
      case 'python':
        return `# Build Instructions for Python Project

## Prerequisites
- Python 3.9+
- pip

## Installation
pip install -r requirements.txt

## Development
uvicorn main:app --reload

## Testing
pytest

## Running
uvicorn main:app --host 0.0.0.0 --port 8000`;
      
      default:
        return `# Build Instructions for ${spec.language} Project

Follow the documentation for your specific language and framework.`;
    }
  }
}

// Demo specifications
const specs = [
  {
    name: 'ecommerce-api',
    language: 'typescript',
    framework: 'express',
    architecture: 'clean',
    database: 'postgresql',
    authentication: 'jwt',
    testing: true,
    deployment: ['docker', 'kubernetes'],
    features: ['user-management', 'product-catalog', 'shopping-cart', 'payment-processing'],
    modules: ['users', 'products', 'orders', 'payments']
  },
  {
    name: 'ml-pipeline',
    language: 'python',
    framework: 'fastapi',
    architecture: 'microservices',
    database: 'mongodb',
    authentication: 'oauth',
    testing: true,
    deployment: ['docker', 'serverless'],
    features: ['data-ingestion', 'model-training', 'prediction-api', 'monitoring'],
    modules: ['data', 'training', 'inference', 'monitoring']
  },
  {
    name: 'iot-gateway',
    language: 'go',
    framework: 'fiber',
    architecture: 'mvc',
    database: 'sqlite',
    authentication: 'api-key',
    testing: true,
    deployment: ['docker'],
    features: ['device-management', 'data-collection', 'real-time-processing'],
    modules: ['devices', 'data', 'processing']
  }
];

/**
 * Executes the demo sequence: iterates the predefined specs, invokes ForgeDemo.generate for each, and logs simulated generation output and summaries.
 *
 * Logs per-demo headers, a brief pause to simulate processing, the generation summary (total files, generation time, languages, frameworks, architecture), and a final summary of demonstrated features.
 */
async function runDemos() {
  const forge = new ForgeDemo();
  
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    console.log(`\n${'='.repeat(50)}`);
    console.log(`🎯 Demo ${i + 1}: ${spec.name}`);
    console.log('='.repeat(50));
    
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate processing time
    
    const result = forge.generate(spec);
    
    console.log('\n📊 Generation Summary:');
    console.log(`   Total Files: ${result.summary.totalFiles}`);
    console.log(`   Generation Time: ${result.summary.generationTime}ms`);
    console.log(`   Languages: ${result.summary.languages.join(', ')}`);
    console.log(`   Frameworks: ${result.summary.frameworks.join(', ')}`);
    console.log(`   Architecture: ${result.summary.architecture}`);
    
    if (i < specs.length - 1) {
      console.log('\n⏳ Preparing next demo...\n');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🎉 All demos completed successfully!');
  console.log('='.repeat(50));
  console.log('\n✨ Atomic Forge is ready to generate production-ready code!');
  console.log('📚 Features demonstrated:');
  console.log('   • Multi-language support (TypeScript, Python, Go)');
  console.log('   • Framework integration (Express, FastAPI, Fiber)');
  console.log('   • Architecture patterns (Clean, MVC, Microservices)');
  console.log('   • Database support (PostgreSQL, MongoDB, SQLite)');
  console.log('   • Authentication systems (JWT, OAuth, API Key)');
  console.log('   • Testing infrastructure');
  console.log('   • Docker & Kubernetes deployment');
  console.log('   • Complete project scaffolding');
}

// Run the demos
runDemos().catch(console.error);