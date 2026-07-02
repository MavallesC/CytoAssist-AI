from django import template

register = template.Library()

@register.filter(name='replace_underscore')
def replace_underscore(value, arg=" "):
    """
    Reemplaza guiones bajos por otro carácter (espacio por defecto).
    """
    if isinstance(value, str):
        return value.replace('_', arg)
    return value
