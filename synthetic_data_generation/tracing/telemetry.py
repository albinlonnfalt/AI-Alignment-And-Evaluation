"""
Simple OpenTelemetry setup for generative AI tracing.
This module provides basic tracing configuration with automatic OpenAI instrumentation.
Supports both console output and Azure Application Insights.
"""

import os
import functools
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.resources import Resource

def setup_tracing(enable_console: bool = True, enable_app_insights: bool = True):
    """
    Initialize OpenTelemetry tracing with OpenAI auto-instrumentation.
    This will automatically trace all OpenAI API calls with generative AI semantic conventions.
    
    Args:
        enable_console: Whether to output traces to console (default: True)
        enable_app_insights: Whether to send traces to Application Insights (default: True)
    """
    
    # Set up the tracer provider with service name
    resource = Resource.create({
        "service.name": "qa-generator",
        "service.version": "1.0.0",
    })
    
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer_provider = trace.get_tracer_provider()
    
    # Set up console exporter if enabled
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        tracer_provider.add_span_processor(console_processor)
        print("✓ Console trace output enabled")
    
    # Set up Azure Application Insights exporter if enabled and configured
    if enable_app_insights:
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if connection_string:
            try:
                azure_exporter = AzureMonitorTraceExporter(
                    connection_string=connection_string
                )
                azure_processor = BatchSpanProcessor(azure_exporter)
                tracer_provider.add_span_processor(azure_processor)
                print("✓ Azure Application Insights tracing enabled")
                print(f"✓ Connection string configured: {connection_string[:50]}...")
            except Exception as e:
                print(f"⚠️  Failed to set up Application Insights: {e}")
                print("   Continuing with console output only...")
        else:
            print("⚠️  APPLICATIONINSIGHTS_CONNECTION_STRING not found in environment")
            print("   Set this environment variable to enable Application Insights tracing")
            print("   Continuing with console output only...")
    
    # Auto-instrument OpenAI SDK - this will automatically trace all OpenAI calls
    # with generative AI semantic conventions
    # .parsed() is not supported in the latest OpenAIInstrumentor. Tracing is implemented manually for those cases
    OpenAIInstrumentor().instrument()
    
    print("✓ OpenTelemetry tracing initialized with OpenAI auto-instrumentation")
    print("✓ Service name set to: qa-generator")

def get_tracer(name: str = "qa-generator"):
    """
    Get a tracer instance for manual instrumentation.
    
    Args:
        name: Name of the tracer (service name)
    
    Returns:
        OpenTelemetry tracer instance
    """
    return trace.get_tracer(name)

def traced(span_name: str = None, record_args: bool = True, record_result: bool = True):
    """
    Simple decorator to automatically trace method calls.
    
    Args:
        span_name: Optional custom span name. If not provided, uses method name.
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value attributes
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the tracer - try from instance first, then default
            if args and hasattr(args[0], 'tracer'):
                tracer = args[0].tracer
            else:
                tracer = get_tracer()
            
            # Determine span name
            if span_name:
                final_span_name = span_name
            else:
                # Auto-generate from class and method
                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__.lower()
                    method_name = func.__name__
                    final_span_name = f"{class_name}.{method_name}"
                else:
                    final_span_name = func.__name__
            
            # Execute with tracing
            with tracer.start_as_current_span(final_span_name) as span:
                try:
                    # Record arguments if requested
                    if record_args:
                        # Get function signature for parameter names
                        import inspect
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        for param_name, value in bound_args.arguments.items():
                            if param_name != 'self':  # Skip self parameter
                                # Convert to string and truncate if too long
                                str_value = str(value)
                                if len(str_value) > 100:
                                    str_value = str_value[:97] + "..."
                                span.set_attribute(f"function.arg.{param_name}", str_value)
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record result if requested
                    if record_result:
                        if hasattr(result, '__len__'):
                            span.set_attribute("function.result.length", len(result))
                        span.set_attribute("function.result.type", type(result).__name__)
                    
                    span.set_attribute("function.success", True)
                    return result
                    
                except Exception as e:
                    # Record exception details
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator
