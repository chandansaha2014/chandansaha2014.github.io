;(function(exports, animationDuration) {
  function tick(el) {
    var card = el.find('.card-carousel__card').last()
    var newCard = card.clone().addClass('new')

    el.prepend(newCard)

    setTimeout(function() {
      newCard.removeClass('new')
    }, 10)

    setTimeout(function() {
      card.remove()
    }, animationDuration)
  }

  function init(el, interval) {
    if (el.length) {
      setInterval(function() {
        tick(el)
      }, interval)
    }
  }

  exports.cardCarousel = {
    init: init
  }
})(window, 500)
;
