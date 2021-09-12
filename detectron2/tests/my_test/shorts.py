def func(function):
    def ii(arg1, arg2):
        print("My arguments are: {0}, {1}".format(arg1, arg2))
        function(arg1, arg2)

    return ii


@func
def cities(city_one, city_two):
    print("Cities I love are {0} and {1}".format(city_one, city_two))


cities("Nairobi", "Accra")