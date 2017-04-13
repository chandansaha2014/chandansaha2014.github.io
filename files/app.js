$(function() {
  function initSlider(el, config) {
    config.onSlideAfter = function() {
      el.stopAuto()
      el.startAuto()
    }

    el.bxSlider(config)
  }

  initSlider($('.bxslider'), {
    infiniteLoop: true,
    controls: false,
    auto: true,
    autoStart: true
  })

  initSlider($('.bxslider-2'), {
    slideWidth: 360,
    minSlides: 1,
    maxSlides: 3,
    slideMargin: 30,
    infiniteLoop: true,
    controls: false,
    auto: true,
    autoStart: true
  })

  cardCarousel.init($('#card-carousel'), 3000)

  $('#mobile-menu-open').on('click', function() {
    $('#mobile-menu').addClass('open')
    $(document.body).addClass('modal-open')
  })

  $('#mobile-menu-close').on('click', function() {
    $('#mobile-menu').removeClass('open')
    $(document.body).removeClass('modal-open')
  })

  window.scrollTo = function(el) {
    $('html, body').animate({
      scrollTop: $(el).offset().top
    }, 1000)
  }
})
;
