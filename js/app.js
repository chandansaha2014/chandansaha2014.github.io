$(document).ready(function() {
    $('#target').teletype({text: ['software engineer.', 'computer scientist.', 
                                'student.', 'curious explorer.', 'sci-fi lover.']});
    $('#cursor').teletype({text: ['|', ' '], delay: 0, pause: 250});
});