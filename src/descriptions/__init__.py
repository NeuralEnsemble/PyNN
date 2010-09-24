import django.template

def fill_django(template, context):
    return django.template.Template(template).render(django.template.Context(context)