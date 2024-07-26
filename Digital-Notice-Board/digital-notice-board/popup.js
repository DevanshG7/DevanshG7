document.addEventListener('DOMContentLoaded', function() {
    const noticeBoard = document.getElementById('notice-board');
  
    // Load widgets
    const widgets = ['googleSpreadsheet', 'googleForm'];
    widgets.forEach(widget => {
      fetch(`widgets/${widget}/${widget}.html`)
        .then(response => response.text())
        .then(data => {
          const widgetElement = document.createElement('div');
          widgetElement.innerHTML = data;
          noticeBoard.appendChild(widgetElement);
        })
        .catch(error => console.error('Error loading widget:', error));
    });

  });
  