"""Comprehensive evaluation framework for RAG system with/without reranker."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from langchain_retrieval_chain import LangChainRetrievalQAChain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAGEvaluator:
    """Evaluate RAG system performance with multiple metrics."""

    def __init__(self):
        self.test_questions = []
        self.ground_truths = []
        self.results_with_reranker = []
        self.results_without_reranker = []

    def load_test_data(self, test_data_path: str) -> None:
        """Load test questions and ground truth answers."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.test_questions = data.get('questions', [])
            self.ground_truths = data.get('ground_truths', [])
            
            logger.info("Loaded %d test questions", len(self.test_questions))
        except Exception as e:
            logger.error("Failed to load test data: %s", e)
            raise

    def generate_test_data_from_documents(self, dataset_dir: str = "Datasets/data") -> None:
        """Generate test questions and ground truths from existing documents."""
        try:
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

            # Generate questions based on document content
            test_data = {
                'questions': [],
                'ground_truths': []
            }

            # Sample questions for each document
            sample_questions = [
                "What are the main concepts in {topic}?",
                "Explain the key principles of {topic}.",
                "What are the main activities involved in {topic}?",
                "Describe the importance of {topic} in software development.",
                "What are the best practices for {topic}?"
            ]

            # Generate questions for each document
            for file_path in sorted(dataset_path.glob("*.txt")):
                if file_path.name.startswith("."):
                    continue

                topic = file_path.stem.replace("_", " ")
                raw_text = file_path.read_text(encoding="utf-8")

                # Generate 2 questions per document
                for i, question_template in enumerate(sample_questions[:2]):
                    question = question_template.format(topic=topic)
                    
                    # Extract relevant ground truth from document
                    # Use first 300 words as ground truth context
                    words = raw_text.split()[:300]
                    ground_truth = " ".join(words)
                    
                    test_data['questions'].append(question)
                    test_data['ground_truths'].append({
                        'answer': ground_truth,
                        'source': str(file_path.name),
                        'key_concepts': self._extract_key_concepts(ground_truth)
                    })

            self.test_questions = test_data['questions']
            self.ground_truths = test_data['ground_truths']
            
            # Save generated test data
            with open('generated_test_data.json', 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Generated %d test questions from documents", len(self.test_questions))

        except Exception as e:
            logger.error("Failed to generate test data: %s", e)
            raise

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using simple heuristics."""
        # Simple keyword extraction - can be enhanced with NLP
        keywords = []
        
        # Look for capitalized terms, nouns, etc.
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Add some common software engineering terms
        common_terms = ['software', 'engineering', 'design', 'testing', 'development', 
                       'requirements', 'implementation', 'quality', 'process', 'methodology']
        
        # Combine and deduplicate
        all_terms = list(set(words + common_terms))
        return all_terms[:10]  # Return top 10

    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation comparing with/without reranker."""
        if not self.test_questions:
            raise ValueError("No test questions loaded. Call load_test_data() or generate_test_data_from_documents() first.")

        # Initialize chains
        chain_with_reranker = LangChainRetrievalQAChain(
            index_path="indexes/faiss_index_all_mini.index",
            use_reranker=True
        )

        chain_without_reranker = LangChainRetrievalQAChain(
            index_path="indexes/faiss_index_all_mini.index", 
            use_reranker=False
        )

        # Run evaluation for each question
        for i, question in enumerate(self.test_questions):
            logger.info("Evaluating question %d/%d: %s", i+1, len(self.test_questions), question[:50] + "...")

            # Get answers from both systems
            try:
                # With reranker
                result_with = chain_with_reranker.answer(question)
                
                # Without reranker  
                result_without = chain_without_reranker.answer(question)
                
                # Store results
                self.results_with_reranker.append({
                    'question': question,
                    'answer': result_with.get('result', ''),
                    'sources': result_with.get('source_documents', [])
                })
                
                self.results_without_reranker.append({
                    'question': question,
                    'answer': result_without.get('result', ''),
                    'sources': result_without.get('source_documents', [])
                })
                
            except Exception as e:
                logger.error("Failed to evaluate question %d: %s", i+1, e)
                continue

        # Calculate metrics
        metrics_with = self._calculate_metrics(self.results_with_reranker)
        metrics_without = self._calculate_metrics(self.results_without_reranker)

        return {
            'with_reranker': metrics_with,
            'without_reranker': metrics_without,
            'comparison': self._compare_metrics(metrics_with, metrics_without)
        }

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate various evaluation metrics."""
        if not results:
            return {}

        total_questions = len(results)
        
        # Initialize metric accumulators
        accuracy_scores = []
        faithfulness_scores = []
        context_recall_scores = []
        answer_quality_scores = []
        
        for i, result in enumerate(results):
            question = result['question']
            answer = result['answer']
            sources = result['sources']
            ground_truth = self.ground_truths[i] if i < len(self.ground_truths) else {}
            
            # Calculate individual metrics
            accuracy = self._calculate_accuracy(answer, ground_truth)
            faithfulness = self._calculate_faithfulness(answer, sources)
            context_recall = self._calculate_context_recall(sources, ground_truth)
            answer_quality = self._calculate_answer_quality(answer)
            
            accuracy_scores.append(accuracy)
            faithfulness_scores.append(faithfulness)
            context_recall_scores.append(context_recall)
            answer_quality_scores.append(answer_quality)

        # Calculate averages
        metrics = {
            'accuracy': np.mean(accuracy_scores) if accuracy_scores else 0.0,
            'faithfulness': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            'context_recall': np.mean(context_recall_scores) if context_recall_scores else 0.0,
            'answer_quality': np.mean(answer_quality_scores) if answer_quality_scores else 0.0,
            'sample_size': total_questions,
            'accuracy_scores': accuracy_scores,
            'faithfulness_scores': faithfulness_scores,
            'context_recall_scores': context_recall_scores,
            'answer_quality_scores': answer_quality_scores
        }

        return metrics

    def _calculate_accuracy(self, answer: str, ground_truth: Dict) -> float:
        """Calculate answer accuracy based on ground truth."""
        if not answer or not ground_truth:
            return 0.0

        gt_answer = ground_truth.get('answer', '')
        key_concepts = ground_truth.get('key_concepts', [])
        
        # Simple accuracy: percentage of key concepts mentioned in answer
        if not key_concepts:
            return 0.0

        mentioned_concepts = 0
        for concept in key_concepts:
            if concept.lower() in answer.lower():
                mentioned_concepts += 1

        accuracy = mentioned_concepts / len(key_concepts)
        return accuracy

    def _calculate_faithfulness(self, answer: str, sources: List[Dict]) -> float:
        """Calculate faithfulness - how well answer is supported by sources."""
        if not answer or not sources:
            return 0.0

        # Extract key claims from answer
        answer_sentences = re.split(r'[.!?]', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return 0.0

        # Check if each claim is supported by any source
        supported_claims = 0
        
        for claim in answer_sentences:
            claim_supported = False
            
            for source in sources:
                source_text = source.page_content.lower()
                claim_lower = claim.lower()
                
                # Check if claim appears in source (with some flexibility)
                if claim_lower in source_text:
                    claim_supported = True
                    break
                
                # Check for key terms matching
                claim_words = set(claim_lower.split())
                source_words = set(source_text.split())
                
                if len(claim_words & source_words) >= 3:  # At least 3 matching words
                    claim_supported = True
                    break
            
            if claim_supported:
                supported_claims += 1

        faithfulness = supported_claims / len(answer_sentences)
        return faithfulness

    def _calculate_context_recall(self, sources: List[Dict], ground_truth: Dict) -> float:
        """Calculate context recall - how well sources cover ground truth."""
        if not sources or not ground_truth:
            return 0.0

        gt_key_concepts = ground_truth.get('key_concepts', [])
        if not gt_key_concepts:
            return 0.0

        # Check which key concepts are covered by retrieved sources
        covered_concepts = 0
        
        for concept in gt_key_concepts:
            concept_covered = False
            
            for source in sources:
                source_text = source.page_content.lower()
                if concept.lower() in source_text:
                    concept_covered = True
                    break
            
            if concept_covered:
                covered_concepts += 1

        context_recall = covered_concepts / len(gt_key_concepts)
        return context_recall

    def _calculate_answer_quality(self, answer: str) -> float:
        """Calculate answer quality based on length, coherence, etc."""
        if not answer:
            return 0.0

        # Simple quality metrics
        length_score = min(len(answer.split()) / 100, 1.0)  # Ideal: 100+ words
        
        # Check for completeness (contains periods, proper structure)
        sentence_count = len(re.split(r'[.!?]', answer))
        structure_score = min(sentence_count / 3, 1.0)  # Ideal: 3+ sentences
        
        # Combined quality score
        quality_score = (length_score * 0.4 + structure_score * 0.6)
        return quality_score

    def _compare_metrics(self, metrics_with: Dict, metrics_without: Dict) -> Dict[str, Any]:
        """Compare metrics between with/without reranker."""
        comparison = {}
        
        for metric_name in ['accuracy', 'faithfulness', 'context_recall', 'answer_quality']:
            if metric_name in metrics_with and metric_name in metrics_without:
                with_value = metrics_with[metric_name]
                without_value = metrics_without[metric_name]
                
                improvement = ((with_value - without_value) / without_value * 100) if without_value > 0 else 0
                
                comparison[metric_name] = {
                    'with_reranker': float(with_value),
                    'without_reranker': float(without_value),
                    'improvement_percent': float(improvement),
                    'better': 'yes' if with_value > without_value else 'no'
                }

        return comparison

    def generate_report(self, results: Dict[str, Any], report_path: str = "evaluation_report.json") -> None:
        """Generate a comprehensive evaluation report."""
        try:
            report = {
                'evaluation_summary': {
                    'timestamp': self._get_current_timestamp(),
                    'test_questions_count': len(self.test_questions),
                    'metrics_computed': list(results['with_reranker'].keys()),
                },
                'results': results,
                'detailed_scores': {
                    'with_reranker': {
                        'accuracy': results['with_reranker']['accuracy_scores'],
                        'faithfulness': results['with_reranker']['faithfulness_scores'],
                        'context_recall': results['with_reranker']['context_recall_scores'],
                        'answer_quality': results['with_reranker']['answer_quality_scores']
                    },
                    'without_reranker': {
                        'accuracy': results['without_reranker']['accuracy_scores'],
                        'faithfulness': results['without_reranker']['faithfulness_scores'],
                        'context_recall': results['without_reranker']['context_recall_scores'],
                        'answer_quality': results['without_reranker']['answer_quality_scores']
                    }
                }
            }

            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info("Evaluation report saved to %s", report_path)
            
            # Print summary
            self._print_evaluation_summary(results)

        except Exception as e:
            logger.error("Failed to generate report: %s", e)
            raise

    def _print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print a human-readable summary of evaluation results."""
        print("\n" + "="*60)
        print("📊 RAG SYSTEM EVALUATION RESULTS")
        print("="*60)
        
        print(f"Test Questions: {len(self.test_questions)}")
        print(f"Evaluation Date: {self._get_current_timestamp()}")
        print()
        
        # Print comparison table
        print("🔬 METRIC COMPARISON (With vs Without Reranker)")
        print("-"*60)
        print(f"{'Metric':<20} {'Without':<15} {'With':<15} {'Improvement'}")
        print("-"*60)
        
        comparison = results['comparison']
        for metric_name, metric_data in comparison.items():
            without_val = metric_data['without_reranker']
            with_val = metric_data['with_reranker']
            improvement = metric_data['improvement_percent']
            better = "✅" if metric_data['better'] == 'yes' else "❌"
            
            print(f"{metric_name:<20} {without_val:<15.3f} {with_val:<15.3f} {improvement:+.1f}% {better}")
        
        print("\n" + "🎯 SUMMARY")
        print("-"*60)
        
        # Count improvements
        improvements = sum(1 for metric in comparison.values() if metric['better'] == 'yes')
        total_metrics = len(comparison)
        
        print(f"Metrics Improved: {improvements}/{total_metrics}")
        
        if improvements > total_metrics / 2:
            print("🎉 Reranker shows significant performance improvements!")
        else:
            print("⚠️  Reranker performance is mixed - consider fine-tuning")
        
        print("="*60)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main evaluation workflow."""
    print("🚀 Starting RAG System Evaluation")
    print("="*50)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    try:
        # Step 1: Generate or load test data
        print("📚 Generating test data from documents...")
        evaluator.generate_test_data_from_documents()
        
        # Step 2: Run evaluation
        print("🔍 Running evaluation...")
        results = evaluator.run_evaluation()
        
        # Step 3: Generate report
        print("📊 Generating evaluation report...")
        evaluator.generate_report(results)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()