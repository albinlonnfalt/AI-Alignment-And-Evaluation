import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import pandas as pd
from openai import AzureOpenAI
from azure.ai.evaluation import (
    evaluate, 
    AzureOpenAIModelConfiguration,
    RelevanceEvaluator,
    GroundednessEvaluator
)
from evaluation.citation_evaluator.citation_evaluator import CitationEvaluator
from evaluation.correct_evaluator.correct_evaluator_evaluator import CorrectEvaluator
from evaluation.completeness_evaluator.completeness_evaluator import CompletenessEvaluator

class EvaluationFactory:
    def __init__(
            self, 
            model_config: AzureOpenAIModelConfiguration,
            output_folder_base: str,
            input_file: str
    ):
        self.model_config = model_config
        self.output_folder_base = output_folder_base
        self.output_folder_full = self._create_output_folder()
        self.input_file = input_file
        self.output_file = self._create_output_file(file_extension='jsonl')
        self.evaluator_config = {
            "api_version": self.model_config["api_version"],
            "azure_endpoint": self.model_config["azure_endpoint"],
            "api_key": self.model_config["api_key"]
        }



    def run_evaluation(self):
        result = evaluate(
            model_config=self.model_config,
            data=self.input_file, # Provide your data here:
            evaluators={
                "relevance": RelevanceEvaluator(model_config=self.model_config),
                "groundedness": GroundednessEvaluator(model_config=self.model_config),
                "citation": CitationEvaluator(model_config_dict=self.evaluator_config),
                "correct": CorrectEvaluator(model_config_dict=self.evaluator_config),
                "completeness": CompletenessEvaluator(model_config_dict=self.evaluator_config)
            },
            # Column mapping:
            evaluator_config={
                "relevance": {
                    "column_mapping": {
                        "query": "${data.question}",
                        "response": "${data.chatbot_answer}"
                    }
                },
                "groundedness": {
                    "column_mapping": {
                        "query": "${data.question}",
                        "context": "${data.chunk_content}",
                        "response": "${data.ground_truth_answer}"
                    }
                },
                "citation": {
                    "column_mapping": {
                        "question": "${data.question}",
                        "chatbot_answer": "${data.chatbot_answer}"
                    }
                },
                "correct": {
                    "column_mapping": {
                        "question": "${data.question}",
                        "chatbot_answer": "${data.chatbot_answer}",
                        "ground_truth_answer": "${data.ground_truth_answer}"
                    }
                },
                "completeness": {
                    "column_mapping": {
                        "question": "${data.question}",
                        "chatbot_answer": "${data.chatbot_answer}",
                        "ground_truth_answer": "${data.ground_truth_answer}"
                    }
                }
            },

            output_path=self.output_file
        )

        # -- Make the result more easily consumable --
     
        df_rows = self._get_dataframe(result)

        df_kpis = self._get_aggregated_kpis(df_rows)

        self._save_to_excel(
            df_rows=df_rows, 
            df_kpis=df_kpis, 
            excel_path=self._create_output_file(file_extension='xlsx')
        )

        return result, df_rows, df_kpis
    
    def _get_aggregated_kpis(self, df_rows):
        """
        Calculate aggregated KPIs from the DataFrame.
        
        Args:
            df_rows: DataFrame containing evaluation results.
        
        Returns:
            A pandas DataFrame with aggregated KPIs ready for Excel export.
        """
        numeric_cols = df_rows.select_dtypes(include='number')
        boolean_cols = df_rows.select_dtypes(include='bool')

        # Create a list to store all KPI rows
        kpi_rows = []
        
        # Add numeric statistics
        if not numeric_cols.empty:
            for stat_name, stat_func in [('mean', numeric_cols.mean()), 
                                        ('std', numeric_cols.std()), 
                                        ('median', numeric_cols.median())]:
                for col_name, value in stat_func.items():
                    kpi_rows.append({
                        'metric': col_name,
                        'statistic': stat_name,
                        'value': value
                    })
        
        # Add boolean statistics
        if not boolean_cols.empty:
            for col_name in boolean_cols.columns:
                kpi_rows.append({
                    'metric': col_name,
                    'statistic': 'percentage_true',
                    'value': boolean_cols[col_name].mean()
                })
        
        # Convert to DataFrame
        kpis_df = pd.DataFrame(kpi_rows)
        
        return kpis_df
    
    def _get_dataframe(self, result_json):
        """
        Convert the evaluation result JSON to a pandas DataFrame.
        Drops unnecessary columns.
        
        Args:
            result_json: The JSON result from the evaluation.
        
        Returns:
            A pandas DataFrame containing the evaluation results.
        """
        df_rows = pd.DataFrame(result_json["rows"])
        cols_to_drop = [col for col in df_rows.columns if col.endswith('_results') or col.endswith('_threshold') or col.endswith('_result') or 'gpt' in col]
        df_rows = df_rows.drop(columns=cols_to_drop)
        return df_rows
    

    def _save_to_excel(self, df_rows, df_kpis, excel_path=None):
        """
        Save evaluation results and KPIs to an Excel file with multiple sheets.
        
        Args:
            result: The evaluation result from run_evaluation()
            excel_path: Optional path for the Excel file. If None, creates one in output_folder.
        
        Returns:
            The path to the created Excel file.
        """

        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Save detailed results
            df_rows.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Save aggregated KPIs
            df_kpis.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Create a summary sheet with pivot table style view
            if not df_kpis.empty:
                summary_pivot = df_kpis.pivot(index='metric', columns='statistic', values='value')
                summary_pivot.to_excel(writer, sheet_name='KPI_Summary')
        
        return excel_path
    
    def _create_output_file(self, file_extension: str):
        """Create an output file path.

        Args:
            file_extension: The file extension for the output file [jsonl, xlsx].

        Returns:
            The output file path.
        """
        output_file_path = f"{self.output_folder_full}/evaluation_results.{file_extension}"
        
        return output_file_path
    
    def _create_output_folder(self):
        """Create the output folder if it doesn't exist."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_full = f"{self.output_folder_base}/evaluation_results_{timestamp}"
        if not os.path.exists(output_folder_full):
            os.makedirs(output_folder_full)

        return output_folder_full
