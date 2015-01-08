"""
Support module for the `describe()` method of many PyNN classes.

If a supported template engine is available on the Python path, PyNN will use
this engine to produce the output from `describe()`. As a fall-back, it will
use the built-in string.Template engine, but this produces much less
well-formatted output, as it does not support hierarchical contexts, loops or
conditionals.

Currently supported engines are Cheetah Template and Jinja2, but it should be
trivial to add others.

If a user has a preference for a particular engine (e.g. if they are
providing their own templates for `describe()`), they may set the
DEFAULT_TEMPLATE_ENGINE module attribute, e.g.::

from pyNN import descriptions
descriptions.DEFAULT_TEMPLATE_ENGINE = 'jinja2'


:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    basestring
except NameError:
    basestring = str
import string
import os.path

DEFAULT_TEMPLATE_ENGINE = None # can be set by user
TEMPLATE_ENGINES = {}


def get_default_template_engine():
    """
    Return the default template engine class.
    """
    default = DEFAULT_TEMPLATE_ENGINE or list(TEMPLATE_ENGINES.keys())[0]
    return TEMPLATE_ENGINES[default]

def render(engine, template, context):
    """
    Render the given template with the given context, using the given engine.
    """
    if template is None:
        return context
    else:
        if engine == 'default':
            engine = get_default_template_engine()
        elif isinstance(engine, basestring):
            engine = TEMPLATE_ENGINES[engine]
        assert issubclass(engine, TemplateEngine), str(engine)
        return engine.render(template, context)
    
    
class TemplateEngine(object):
    """
    Base class.
    """
    
    @classmethod
    def get_template(cls, template):
        """
        template may be either a string containing a template or the name of a
        file (relative to pyNN/descriptions/templates/<engine_name>)
        """
        raise NotImplementedError()
        
    @classmethod
    def render(cls, template, context):
        """
        Render the template with the context.
        
        template may be either a string containing a template or the name of a
        file (relative to pyNN/descriptions/templates/<engine_name>)
        
        context should be a dict.
        """
        raise NotImplementedError()


class StringTemplateEngine(TemplateEngine):
    """
    Interface to the built-in string.Template template engine.
    """
    template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'string')
    
    @classmethod
    def get_template(cls, template):
        """
        template may be either a string containing a template or the name of a
        file (relative to pyNN/descriptions/templates/string/)
        """
        template_path = os.path.join(cls.template_dir, template)
        if os.path.exists(template_path):
            f = open(template_path, 'r')
            template = f.read()
            f.close()
        return string.Template(template)
        
    @classmethod
    def render(cls, template, context):
        """
        Render the template with the context.
        
        template may be either a string containing a template or the name of a
        file (relative to pyNN/descriptions/templates/string/)
        
        context should be a dict.
        """
        template = cls.get_template(template)
        return template.safe_substitute(context)
    
TEMPLATE_ENGINES['string'] = StringTemplateEngine


try:
    import jinja2

    class Jinja2TemplateEngine(TemplateEngine):
        """
        Interface to the Jinja2 template engine. 
        """
        env = jinja2.Environment(loader=jinja2.PackageLoader('pyNN.descriptions', 'templates/jinja2'))
        
        @classmethod
        def get_template(cls, template):
            """
            template may be either a string containing a template or the name of a
            file (relative to pyNN/descriptions/templates/jinja2/)
            """
            assert isinstance(template, basestring)
            try: # maybe template is a file
                template = cls.env.get_template(template)
            except Exception: # interpret template as a string
                template = cls.env.from_string(template)
            return template
        
        @classmethod
        def render(cls, template, context):
            """
            Render the template with the context.
            
            template may be either a string containing a template or the name of a
            file (relative to pyNN/descriptions/templates/jinja2/)
            
            context should be a dict.
            """
            template = cls.get_template(template)
            return template.render(context)
        
    TEMPLATE_ENGINES['jinja2'] = Jinja2TemplateEngine
except ImportError:
    pass


try:
    import Cheetah.Template

    class CheetahTemplateEngine(TemplateEngine):
        """
        Interface to the Cheetah template engine.
        """
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'cheetah')
        
        @classmethod
        def get_template(cls, template):
            """
            template may be either a string containing a template or the name of a
            file (relative to pyNN/descriptions/templates/cheetah)
            """
            template_path = os.path.join(cls.template_dir, template)
            if os.path.exists(template_path):
                return Cheetah.Template.Template.compile(file=template_path)
            else:
                return Cheetah.Template.Template.compile(source=template)
    
        @classmethod
        def render(cls, template, context):
            """
            Render the template with the context.
            
            template may be either a string containing a template or the name of a
            file (relative to pyNN/descriptions/templates/cheetah/)
            
            context should be a dict.
            """
            template = cls.get_template(template)(namespaces=[context])
            return template.respond()

    TEMPLATE_ENGINES['cheetah'] = CheetahTemplateEngine
except ImportError:
    pass